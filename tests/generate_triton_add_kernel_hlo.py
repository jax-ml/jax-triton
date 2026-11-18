# Copyright 2026 The jax_triton Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates the HLO for a simple JAX-Triton add kernel, used for testing in XLA."""

import functools
from absl import app
from absl import flags
import jax
from jax import export
from jax.interpreters import mlir
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "Optional path to write the generated HLO text file. If omitted, prints"
    " to stdout.",
)

# Register lowering rule for platform=None so deviceless lowering produces CUDA
# custom call HLO.
mlir.register_lowering(
    jt.triton_lib.triton_kernel_call_p,
    functools.partial(
        jt.triton_lib.triton_kernel_call_lowering,
        jt.triton_lib.make_cuda_target,
    ),
    platform=None,
)


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  x = tl.load(x_ptr + offsets)
  y = tl.load(y_ptr + offsets)
  output = x + y
  tl.store(output_ptr + offsets, output)


def add(x, y):
  return jt.triton_call(
      x,
      y,
      kernel=add_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      grid=(x.size // 1024, 1, 1),
      BLOCK_SIZE=1024,
      # Use a low enough compute capability to ensure it can run in all the
      # GPUs we test against in XLA.
      compute_capability=60,
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  x = jax.ShapeDtypeStruct((1024,), dtype=jnp.float32)
  y = jax.ShapeDtypeStruct((1024,), dtype=jnp.float32)

  # Use jax.export to export the function
  exported = export.export(
      jax.jit(add),
      disabled_checks=[
          export.DisabledSafetyCheck.custom_call("triton_kernel_call_ffi"),
      ],
  )(x, y)

  # Extract HLO text from the exported computation call
  hlo_text = (
      jax.jit(exported.call)
      .lower(x, y)
      .compiler_ir(dialect="hlo")
      .as_hlo_text()
  )

  if _OUTPUT_PATH.value:
    with open(_OUTPUT_PATH.value, "w") as f:
      f.write(hlo_text)
    print(f"Successfully wrote optimized HLO to {_OUTPUT_PATH.value}")
  else:
    print(hlo_text)


if __name__ == "__main__":
  app.run(main)
