# 0.5.0
## Breaking changes
- `float` now follows upstream convention and is represented as `fp32`, instead of old
`fp64`
- `zeroed_outputs=` parameter of `triton_call()` no longer supports zeroing of aliased
input-output arguments.

## New features / bugfixing
- all possible backend initialization options is now fully supported and is handled
similarly to the upstream (via single `kwargs` dictionary).
- support for `@jt.kernel` decorator and a concise Triton-native form of launching a
kernel with `kernel[grid](*args, **kwargs)` syntax.
- arrays and other run-time values can now also be passed as a key-value pair to the
launcher when `out_names=` is set or if a new dictionary form of `out_shape=` is used.
- handling of kernel arguments specialization and defaults values is now fully delegated
to the upstream Triton code, which enables full support for default values, kernel
parameter annotations, related `@triton.jit()` arguments such as `do_not_specialize`,
and also using tuples (including deeply nested), callables or strings as kernel
arguments.
- `out_shape`, `input_output_aliases` and `zeroed_outputs` handling is fully reworked
to support nested tuples and now is based on a kernel signature coordinate system,
instead of flat array indices, leading to a much clearer launcher syntax.
- dictionary form of `input_output_aliases=` is deprecated, but is still fully supported
- `CAN_USE_TRITON` guard is dropped due to obsolescence
- tests grew 187 to 438 test cases


# 0.4.0
- Add support for Gluon kernels
- fixed handling of `TRITON_CACHE_DIR` in line with the upstream treatment

# 0.3.1
- improve in-out parameters handling, getting rid of mandatory aliased parameters in
kernel's signature
- revamping and bugfixing to support `jax>=0.8.2`
- monkeypatching Triton's `HIPBackend` to get rid of unnecessary dependency on `torch`