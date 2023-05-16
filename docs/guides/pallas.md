# Writing custom kernels using Pallas, a JAX kernel sublanguage

Pallas allows you to write your own custom kernels using JAX directly!

## Your first Pallas kernel: `add_one`

Let's try to write a kernel that adds one to a vector. We'll first do some basic imports.
```python
from functools import partial
import jax.numpy as jnp
import numpy as np
import jax

from jax_triton import pallas as pl
```

First we'll write a kernel. A kernel is a program that will be executed (potentially multiple times) on an accelerator. Our `add_one_kernel` function should read from inputs, perform the computation, then write to the outputs.
```python
def add_one_kernel(x_ref, o_ref, *, block_size: int):
  i = pl.program_id(0)
  offsets = i * block_size + jnp.arange(block_size)
  o_ref[offsets] = x_ref[offsets] + 1
```
We perform indexed reads from and in-place indexed writes to `Ref`s using NumPy-style indexing.

We now write a JAX function that runs the kernel using `pallas_call`. The `grid` argument indicates how many times the kernel will be invoked.
```python
@jax.jit
def add_one(x):
  return pl.pallas_call(
      partial(add_one_kernel, block_size=8),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      grid=pl.cdiv(x.shape[0], 8))(x)

```

We can now call this JAX function like any other.
```python
x = jnp.arange(32)
np.testing.assert_allclose(add_one(x), x + 1)
```
We can also even `jax.vmap` it!
```python
x = jnp.arange(4 * 32).reshape((4, 32))
np.testing.assert_allclose(jax.vmap(add_one)(x), x + 1)
```

## Pallas programming guide

We'll now cover some in writing Pallas kernels.

### Launching programs in a *grid*

Next, we'll write a *kernel*. A *kernel* is a JAX function that takes in `Ref` objects (mutable JAX types) corresponding to inputs and outputs. In this case, we'll have one `Ref` for the input (`x_ref`) and one for the output (`o_ref`).

Conceptually, this kernel function will be executed multiple times, each on a different chunk, or block, of the inputs and outputs. We'll parameterize our kernel by a static integer `block_size`, which will determine the size of the "chunks" or "blocks" of our input that each instance of the kernel will operate on.

<style>
td {
    border: 1px solid black;
    font-size: 1.2em;
    background-color: var(--md-default-bg-color);
    color: var(--md-fg-color--lightest);
}
th {
    border: 1px solid black;
    font-size: 1.2em;
    color: var(--md-fg-color--lightest);
    background-color: var(--md-accent-fg-color);
}
</style>
<table>
<th colspan=8 style="text-align: center;">Program indices</th>
<tr>
<td>0</td>
<td>1</td>
<td>2</td>
<td>3</td>
<td>4</td>
<td>5</td>
<td>6</td>
<td>7</td>
</tr>
</table>
