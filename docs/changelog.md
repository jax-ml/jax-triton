# Changelog

<!--
Remember to align the itemized text with the first line of an item within a list.
-->

## jax-triton 0.4.0 (unreleased)

* New features
  * Added support for Gluon kernels.
  * `triton_call` now accepts pytrees as inputs and outputs. Note that
    `input_output_aliases` and `zeroed_outputs` now index into the *flattened*
    inputs and outputs, matching `jax.experimental.pallas.pallas_call`.
  * `triton_call` now accepts arbitrary backend options via the
    `backend_options` argument.
  * Added a `has_side_effect` argument to `triton_call` to prevent dead-code
    elimination (DCE) at the XLA level.
  * Missing metaparams are now filled in from the kernel parameter defaults.
  * `jax_triton` is now importable on hosts without GPU support; the GPU support
    check is deferred until lowering time.

* Breaking changes
  * Removed the `jax_triton.utils` submodule together with `jax_triton.cdiv` and
    `jax_triton.next_power_of_2`. Use `triton.cdiv` and
    `triton.next_power_of_2` instead.
  * Removed the `enable_fp_fusion` argument to `triton_call`. Pass
    `backend_options=dict(enable_fp_fusion=...)` instead.
  * When using `input_output_aliases`, aliased buffers must no longer be
    declared as implicit outputs; the aliased input buffer is used directly. A
    clearer error is now raised for the old calling convention.

* Bug fixes
  * Fixed `input_output_aliases` indexing when positional scalar arguments
    precede array inputs.
  * Temporary files are no longer leaked when compiling on ROCm.
  * The compute capability is now resolved from `gpu_info` when a device is
    unavailable.
  * `jax_triton` now unsets `TRITON_CACHE_DIR` on import for compatibility
    with Triton >= 3.4.0.
