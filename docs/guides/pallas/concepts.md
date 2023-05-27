# Concepts

We'll now cover some important concepts in Pallas.

## Single-program, multiple data (SPMD)

Pallas has a [single-program, multiple data (SPMD)](https://en.wikipedia.org/wiki/Single_program,_multiple_data) programming paradigm. This means you write a single function that describes your computation and it'll be executed many times with different inputs. On a GPU, this bottoms out in the function being executed in parallel over many threads.

If you're familiar with JAX, you may have seen the term SPMD before. In fact, it is the programming model for `jax.pmap` and `jax.experimental.shard_map`! However, in those cases, we parallelize computations over many different accelerators. In Pallas, we want to parallelize computations *within* an accelerator.


## Launching kernels in a "grid"

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

## Stateful JAX (i.e. working with `Ref`s)

How do we actually write our kernels? A Pallas kernel is a function that operates on arrays in "fast memory", i.e memory that is very close to our compute (on GPUs, this corresponds to L1 caches). In Pallas, we explicitly control how we interact with this memory -- specifically, we control where/when we load and store to and from memory.

This level of control over memory isn't available in vanilla JAX. JAX doesn't even offer mutation semantics at all! However, JAX recently added support for a *state* side-effect via mutable references to arrays. Pallas repurposes `Ref`s to explicitly control how memory is accessed within kernels.

When we write our kernel, it will take `Ref`s, not JAX arrays, as inputs. In addition, we will have `Ref`s for the outputs that we are responsible for writing the final values to.

`Ref`s, or "references", are mutable wrappers around Array values. A common pattern when working with `Ref`s is to write functions that take in both "input" and "output" `Ref`s. Usually you read from the input `Ref`s and write into the output `Ref`s.

For example, here is a function that reads from an input `Ref` and copies its value to an output `Ref`:
```python
def identity_stateful(input_ref, output_ref):
  value = input_ref[...]  # Uses NumPy-like indexing semantics to read values
  output_ref[...] = value  # Writes value to `output_ref`
```

!!! note
    The `[...]` notation corresponds to reading the *entire* value of the `Ref`.

Here's a function that computes `exp(x)`.
```python
def exp_stateful(input_ref, output_ref):
  output_ref[...] = jnp.exp(input_ref[...])
```

You can also read from and write to the same `Ref`.
```python
def exp_plus_one_stateful(input_ref, output_ref):
  output_ref[...] = jnp.exp(input_ref[...])  # Read from Ref
  output_ref[...] = output_ref[...] + 1  # Read from and write to Ref
```
Conceptually, `exp_plus_one_stateful` updates `output_ref` in-place to compute `exp(x) + 1`.

On GPU, when we read from a `Ref`, we are loading values from GPU HBM into GPU shared memory and conversely when we are writing to a `Ref`, we are writing into GPU HBM.
