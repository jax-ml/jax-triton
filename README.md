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
tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html):

```python
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    length,
    out_ptr,                   # out_shape pointers follow the inputs.
    block_size: tl.constexpr,  # constexpr params can go anywhere.
):
  """Adds two vectors output = x + y."""
  pid = tl.program_id(axis=0)
  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)
  mask = offsets < length
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
```

Then we can apply it to JAX arrays using `jax_triton.triton_call`:

```python
import jax
import jax.numpy as jnp
import jax_triton as jt

def add(x: jax.Array, y: jax.Array) -> jax.Array:
  block_size = 8
  return jt.triton_call(
      x,
      y,
      x.size,
      kernel=add_kernel,
      out_shape=jax.typeof(x),
      grid=(x.size // block_size,),
      block_size=block_size)

x_val = jnp.arange(8)
y_val = jnp.arange(8, 16)
print(add(x_val, y_val))
print(jax.jit(add)(x_val, y_val))
```

One could also use in-out parameters for kernels by passing a read-write
`Ref` (created via `jax.new_ref`). The kernel mutates the `Ref` in place, so it
is not listed in `out_shape`:

```python

@triton.jit
def add_inplace_y_kernel(
    x_ptr,          # input vector
    y_inout_ptr,    # explicit in-out vector (could be anywhere)
    length,
    block_size: tl.constexpr,
):
  """Adds two vectors in place: y = x + y."""
  pid = tl.program_id(axis=0)
  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)
  mask = offsets < length
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_inout_ptr + offsets, mask=mask)
  output = x + y
  tl.store(y_inout_ptr + offsets, output, mask=mask)


# jitting isn't mandatory, but makes invocation more efficient.
@jax.jit
def add_inplace_y(x: jax.Array, y_ref) -> None:
  block_size = 8
  jt.triton_call(
      x,
      y_ref,         # read-write Ref argument, mutated in place
      x.size,
      kernel=add_inplace_y_kernel,
      out_shape=(),  # no allocated outputs; the Ref is mutated in place
      grid=(x.size // block_size,),
      block_size=block_size)

x_val = jnp.arange(8)
y_ref = jax.new_ref(jnp.arange(8, 16))
add_inplace_y(x_val, y_ref)
print(y_ref[...])
```

See [the examples
directory](https://github.com/jax-ml/jax-triton/tree/main/examples), especially
[fused_attention.py](https://github.com/jax-ml/jax-triton/blob/main/examples/fused_attention.py)
and [the fused attention
ipynb](https://github.com/jax-ml/jax-triton/blob/main/examples/JAX_%2B_Triton_Flash_Attention.ipynb).

Some other use-cases are also covered in [tests](https://github.com/jax-ml/jax-triton/tree/main/tests).

## Installation

```bash
$ pip install jax-triton
```

Make sure you have a CUDA- or ROCm- compatible `jax` installed. For example you could run:
```bash
$ pip install "jax[cuda12]"
```

## Development

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
