# Calling Triton kernels from JAX

The primary way of using JAX Triton is using `jax_triton.triton_call` to call handwritten Triton kernels
from inside JIT-ted JAX programs.

::: jax_triton.triton_call
    options:
      show_root_heading: true
      show_source: false
      show_signature: false
