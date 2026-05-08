# jax-triton

![PyPI version](https://img.shields.io/pypi/v/jax-triton)

The `jax-triton` repository contains integrations between [JAX](https://github.com/jax-ml/jax)
and [Triton](https://github.com/openai/triton), including support for Gluon dialect.

Documentation can be found [here](https://jax-ml.github.io/jax-triton).

*This is not an officially supported Google product.*

## Quickstart

The main function of interest is `jax_triton.triton_call` for applying Triton
functions to JAX arrays, including inside `jax.jit`-compiled functions. For
example, we can define [a kernel from the Triton
tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py):

```python
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,        # First 3 arguments
    y_ptr,        # are input
    length,       # arguments.
    output_ptr,   # Implicit output argument goes after inputs.
    block_size: tl.constexpr, # Constexpr params go last.
):
  """Adds two vectors output = x + y."""
  pid = tl.program_id(axis=0)
  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)
  mask = offsets < length
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)
```

Then we can apply it to JAX arrays using `jax_triton.triton_call`:

```python
import jax
import jax.numpy as jnp
import jax_triton as jt

def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  block_size = 8
  return jt.triton_call(
      x,                  # Kernel's input arguments are the first
      y,                  # in jt.triton_call(). The output argument
      x.size,             # is passed implicitly.
      kernel=add_kernel,
      out_shape=x,
      grid=(x.size // block_size,),
      block_size=block_size   # Constexpr params are passed as kwargs
    )

x_val = jnp.arange(8)
y_val = jnp.arange(8, 16)
print(add(x_val, y_val))
print(jax.jit(add)(x_val, y_val))
```

Or better, we could use `@jt.kernel` decorator, capable of capturing `triton_call()`
arguments in advance and allowing to use a familiar native Triton's
`kernel[grid](*args, **kwargs)` syntax. Here's how to use it to cache kernel's
input-output parameter specification:

```python
@jt.kernel(input_output_aliases="y_inout_ptr")  # this tells triton_call() launcher that
# argument of `y_inout_ptr` parameter is an in-out array
@triton.jit
def add_inplace_y_kernel(
    x_ptr,          # input vector
    y_inout_ptr,    # explicit in-out vector (could be anywhere)
    length,
    block_size: tl.constexpr,
):
  """Adds two vectors output = x + y."""
  pid = tl.program_id(axis=0)
  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)
  mask = offsets < length
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_inout_ptr + offsets, mask=mask)
  output = x + y
  tl.store(y_inout_ptr + offsets, output, mask=mask)


from functools import partial

# jitting or jitting with donation isn't mandatory, but makes invocation more efficient.
# Otherwise XLA would have to make a copy of each non-donated in-out argument before
# calling a kernel, since JAX arrays are immutable by default.
@partial(jax.jit, donate_argnames="y")
def add_inplace_y(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  block_size = 8
  grid = x.size // block_size
  return add_inplace_y_kernel[grid](x, y, x.size, out_shape=x, block_size=block_size)

x_val = jnp.arange(8)
y_val = jnp.arange(8, 16)
print(add_inplace_y(x_val, y_val))
```

Note that you could use advanced Triton features such as passing strings, tuples and
callables, for example:

```python
from jax import random
import numpy as np
from triton.language.extra import libdevice
from typing import NamedTuple
import time

class Function(NamedTuple):
  fn: tl.constexpr
  captured: tuple

@triton.jit
def func1(x_ptr, y_ptr: tl.const, SIZE: tl.constexpr):
  off = tl.arange(0, SIZE)
  x = tl.load(x_ptr + off)
  y = tl.load(y_ptr + off)
  x1 = libdevice.sin(x)
  x2 = libdevice.cos(x)
  z = x1 * x1 + x2 * x2
  tl.store(x_ptr + off, z)
  y = libdevice.asin(y) + libdevice.acos(y)
  return z, libdevice.floor(2 * y)

@triton.jit
def floor_of_func(values, SIZE: tl.constexpr, FUNC_NAME: tl.constexpr):
  off = tl.arange(0, SIZE)
  return libdevice.floor(getattr(libdevice, FUNC_NAME)(values))

@triton.jit
def aggregate(Ptrs):
  z = tl.zeros([], tl.float32)
  for i in tl.static_range(len(Ptrs)):
    z += Ptrs[i]
  return z

@jt.kernel
@triton.jit
def kernel(capture, out_ptr, SIZE: tl.constexpr, FUNC_NAME: tl.constexpr):
  off = tl.arange(0, SIZE)
  t1, t2 = capture.fn(*capture.captured, SIZE=SIZE)
  t3 = floor_of_func(t1, SIZE=SIZE, FUNC_NAME=FUNC_NAME)
  t4 = t2 * t3
  result = aggregate((t4, t4 * t4)).to(tl.int32)
  tl.store(out_ptr + off, result)

size = 8
k1, k2 = random.split(random.key(time.perf_counter_ns()), 2)
x = random.uniform(k1, (size,), dtype=jnp.float32)
y = random.uniform(k2, (size,), dtype=jnp.float32)

fn = Function(func1, (x, y))
out, x = kernel[(1,)](
  fn,  # essentially a tuple of (func_name, (2 arrays in a subtuple))
  SIZE=size,
  FUNC_NAME="exp",
  out_shape=jnp.zeros(size, dtype=jnp.int32),
  input_output_aliases=("capture", 1, 0),  # a path inside `capture` argument, 0th
  # element of 1st subtuple, i.e. `x`. Note that since this is a tuple, it
  # references just a single element (or all its embedded arrays if it's a tuple,
  # but in this invocation it references array `x`)
)
np.testing.assert_array_equal(out, jnp.full((size,), 42, dtype=jnp.int32))
assert out.dtype == jnp.int32
np.testing.assert_allclose(x, jnp.full((size,), 1.0, dtype=jnp.float32), rtol=5e-7)
assert x.dtype == jnp.float32
```

For detailed information on `triton_call()` parameters, refer to its docstring.

See [the examples
directory](https://github.com/jax-ml/jax-triton/tree/main/examples), especially
[fused_attention.py](https://github.com/jax-ml/jax-triton/blob/main/examples/fused_attention.py)
and [the fused attention
ipynb](https://github.com/jax-ml/jax-triton/blob/main/examples/JAX_%2B_Triton_Flash_Attention.ipynb).

Many use-cases are covered in [tests](https://github.com/jax-ml/jax-triton/tree/main/tests).

## Installation

```bash
$ pip install jax-triton
```

Make sure you have a CUDA- or ROCm- compatible `jax` installed. For example you could run:
```bash
$ pip install "jax[cuda12]"
```

## Development / Bleeding edge version

To develop `jax-triton`, you can clone the repo with:
```bash
$ git clone https://github.com/jax-ml/jax-triton.git
```
and do an editable install with:
```bash
$ cd jax-triton
$ pip install -e .
```
To run the `jax-triton` tests, you'll need `pytest`:
```bash
$ pip install pytest
$ pytest tests/
```

## Known limitations (a non-exhaustive list)

0. Due to JAX's custom call API restrictions, purely output parameters of a kernel
should be defined last in the kernel's signature (ignoring constexpr's). Certain
relaxations of this rule exist, but it's just simpler to always follow it by reordering
kernel parameters.

1. Be aware that a benign in the upstream Triton pattern of pre-allocating an empty
buffer and passing it as `jax-triton`'s input-output buffer just to be only written to
by the kernel, might have a cost of host -> device data copying in JAX. Use purely
output arguments instead.

2. Autotuner / heuristics might have support gaps, for example: `Config.pre_hook` /
`Config.post_hook` hooks aren't implemented; custom `do_bench` / `perf_model` /
`early_config_prune` / `top_k` on `triton.autotune` might not behave correctly.

3. Triton Python-runtime features, such as `JITFunction.warmup`, `.preload` and others
such as Interpreter mode, aren't supported.

4. Might not work well on systems with several different GPU models.

5. JAX-level transformations, such as `jax.grad`, `jax.jvp`, `jax.vjp` aren't supported
on kernels.

