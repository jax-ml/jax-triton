# 0.5.0
## Breaking changes
- `float` now follows upstream convention and is represented as `fp32`, instead of the
old `fp64`
- `zeroed_outputs=` parameter of `triton_call()` no longer supports zeroing of aliased
input-output arguments.

## New features / bug fixes
- all possible backend initialization options are now fully supported and handled
similarly to upstream (via a single `kwargs` dictionary).
- support for `@jt.kernel` decorator and a concise Triton-native form of launching a
kernel with `kernel[grid](*args, **kwargs)` syntax.
- `out_names=` triton_call() parameter introduced to let specify output-only parameters
in the kernel's signature. Alternatively, a new dictionary form of `out_shape=` is
introduced for the same purpose.
- arrays and other runtime values can now also be passed as key-value arguments to the
launcher when `out_names=` is set or if a new dictionary form of `out_shape=` is used.
- handling of kernel argument specialization and default values is now fully delegated
to the upstream Triton code, which enables full support for default values, kernel
parameter annotations, related `@triton.jit()` arguments such as `do_not_specialize`,
and also using tuples (including deeply nested ones), callables, or strings as kernel
arguments.
- `out_shape`, `input_output_aliases` and `zeroed_outputs` handling is fully reworked
to support nested tuples and is now based on a kernel signature coordinate system,
instead of flat array indices, leading to a much clearer launcher syntax.
- dictionary form of `input_output_aliases=` is deprecated, but is still fully supported
- `CAN_USE_TRITON` guard dropped due to obsolescence
- test cases grew from 187 to 438


# 0.4.0
- Add support for Gluon kernels
- Fix handling of `TRITON_CACHE_DIR` in line with the upstream treatment

# 0.3.1
- Improve in-out parameter handling, getting rid of mandatory aliased parameters in
kernel's signature
- Revamp and fix bugs to support `jax>=0.8.2`
- Monkey-patch Triton's `HIPBackend` to get rid of unnecessary dependency on `torch`