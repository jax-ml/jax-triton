# jax-triton

![PyPI version](https://img.shields.io/pypi/v/jax-triton)

The `jax-triton` repository contains integrations between [JAX](https://github.com/google/jax) and [Triton](https://github.com/openai/triton).

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
    x_ptr,
    y_ptr,
    length,
    output_ptr,
    block_size: tl.constexpr,
):
  """Adds two vectors."""
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
  out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
  block_size = 8
  return jt.triton_call(
      x,
      y,
      x.size,
      kernel=add_kernel,
      out_shape=out_shape,
      grid=(x.size // block_size,),
      block_size=block_size)

x_val = jnp.arange(8)
y_val = jnp.arange(8, 16)
print(add(x_val, y_val))
print(jax.jit(add)(x_val, y_val))
```

See [the examples
directory](https://github.com/jax-ml/jax-triton/tree/main/examples), especially
[fused_attention.py](https://github.com/jax-ml/jax-triton/blob/main/examples/fused_attention.py)
and [the fused attention
ipynb](https://github.com/jax-ml/jax-triton/blob/main/examples/JAX_%2B_Triton_Flash_Attention.ipynb).

## Installation

```bash
$ pip install jax-triton
```

You can either use a stable release of `triton` or a nightly release.

Make sure you have a CUDA-compatible `jax` installed. For example you could run:
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
