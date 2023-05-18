We'll now cover some important concepts when writing your Pallas kernels.

## Programming model

### Single-program, multiple data (SPMD)

Pallas has a [single-program, multiple data (SPMD)](https://en.wikipedia.org/wiki/Single_program,_multiple_data) programming paradigm. This means you write a single function that describes your computation and it'll be executed many times with different inputs. On a GPU, this bottoms out in the function being executed in parallel over many threads.

If you're familiar with JAX, you may have seen the term SPMD before. In fact, it is the programming model for `jax.pmap` and `jax.experimental.shard_map`! However, in those cases, we parallelize computations over many different accelerators. In Pallas, we want to parallelize computations *within* an accelerator.


### Launching kernels in a "grid"

In Pallas, after we write our SPMD function, which we'll call a **kernel**, we'll need to specify how we execute our kernel using a **grid**. A grid is a tuple of integers that specifies how many times we'll invoke the kernel.

!!! info "What's a grid, more specifically?"

    If we think of this grid as a "shape" (think NumPy array or JAX array shape), it encodes a set of possible indices. For example, the grid `(8,)` encodes the indices `[(0,), (1,), (2,), ... , (7,)]` and the grid `(1, 2, 3)` encodes the indices `[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2)]`.

The Pallas grid behaves much like the CUDA grid in that we'll be executing our kernel once per index in the grid.

```
# Pseudocode for how kernels are executed
for ndindex in np.ndenumerate(grid):
  run_kernel(ndindex, kernel)
```
However, on GPUs, this for loop will be *parallelized*.

Each instance of the kernel, which we'll call a **program**, can select out which part of our input data it will operate on using its `program_id`, i.e. which index in the grid it corresponds to.

### Stateful JAX (i.e. working with `Ref`s)

How do we actually write our kernels? A Pallas kernel is a function that operates on arrays in "fast memory", i.e memory that is very close to our compute (on GPUs, this corresponds to L1 caches). In Pallas, we explicitly control how we interact with this memory -- specifically, we control where/when we load and store to and from memory.

This level of control over memory isn't available in vanilla JAX. JAX doesn't even offer mutation semantics at all! However, JAX recently added support for a *state* side-effect via mutable references to arrays. Pallas repurposes `Ref`s to explicitly control how memory is accessed within kernels.

When we write our kernel, it will take `Ref`s, not JAX arrays,  as inputs. In addition, we will have `Ref`s for the outputs that we are responsible for writing the final values to.

For example, here is a kernel that reads from an input `Ref` and copies its value to an output `Ref`:
```python
def identity_kernel(input_ref, output_ref):
  value = input_ref[...]  # Uses NumPy-like indexing semantics to read values
  output_ref[...] = value  # Writes value to `output_ref`
```

!!! warning
    When writing to `Ref`s, remember that we might be executing our kernel in parallel, so there may be race conditions when writing to the same location in memory.

<!-- Next, we'll write a *kernel*. A *kernel* is a JAX function that takes in `Ref` objects (mutable JAX types) corresponding to inputs and outputs. In this case, we'll have one `Ref` for the input (`x_ref`) and one for the output (`o_ref`).

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
</table> -->
