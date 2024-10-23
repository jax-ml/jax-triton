# JAX-Triton documentation

JAX-Triton is a repository containing containing integrations between [JAX](https://github.com/google/jax)
and [Triton](https://github.com/openai/triton).

JAX is a Python library for accelerated numerical computing and Triton is a Python library and compiler for writing custom GPU kernels.
When we put the two together, we get JAX-Triton, which enables writing custom GPU kernels using Triton that can be embedded inside of JAX programs.

## Getting started

### Installing JAX-Triton

You can install JAX-Triton with `pip`. This will also install a compatible JAX and Triton.
```bash
$ pip install jax-triton
```

JAX-Triton only works with JAX on GPU, so you'll need to make sure you have a CUDA-compatible `jaxlib` installed.
For example you could run:
```bash
$ pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Check out the [JAX installation guide](https://github.com/google/jax#pip-installation-gpu-cuda) for details.

### Installation at HEAD

JAX-Triton is developed at JAX and jaxlib HEAD and close to Triton HEAD. To get
a bleeding edge installation of JAX-Triton, run:

```bash
$ pip install 'jax-triton @ git+https://github.com/jax-ml/jax-triton.git'
```

This should install compatible versions of JAX and Triton.

JAX-Triton requires jaxlib with GPU support. You could install the latest stable
release via

```bash
$ pip install jaxlib[cuda12]
```

In rare cases JAX-Triton might need a nighly version of jaxlib. You can install
it following the instructions
[here](https://jax.readthedocs.io/en/latest/installation.html#jax-nightly-installation).

### Quickstart

The main function of interest is `jax_triton.triton_call` for applying Triton
functions to JAX arrays, including inside `jax.jit`-compiled functions. For
example, we can define [a kernel from the Triton
tutorial](https://triton-lang.org/master/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py):

```python
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    block_size: tl.constexpr,
):
  """Adds two vectors."""
  pid = tl.program_id(axis=0)
  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)
  mask = offsets < 8
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
