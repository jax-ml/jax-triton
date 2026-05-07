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

"""Minimal jax-triton example using a host-side TMA TensorDescriptor.

Run with:

  python examples/tma_copy.py

This requires CUDA TMA support, which is available on NVIDIA GPUs with compute
capability >= 90.
"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl


@triton.jit
def tma_copy_kernel(
    x_desc,
    output_ptr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  start_m = pid_m * BLOCK_M
  start_n = pid_n * BLOCK_N

  tile = x_desc.load([start_m, start_n])

  rows = start_m + tl.arange(0, BLOCK_M)
  cols = start_n + tl.arange(0, BLOCK_N)
  output_offsets = rows[:, None] * N_COLS + cols[None, :]
  tl.store(output_ptr + output_offsets, tile)


def require_tma_capable_gpu() -> None:
  try:
    compute_capability = jt.get_compute_capability(0)
  except Exception as exc:
    raise RuntimeError(
        "This example requires jaxlib with CUDA GPU support."
    ) from exc

  if compute_capability < 90:
    raise RuntimeError(
        "This example requires a CUDA GPU with compute capability >= 90; "
        f"device 0 has {compute_capability}."
    )


def tma_copy(x: jax.Array) -> jax.Array:
  if x.ndim != 2:
    raise ValueError(f"expected a rank-2 input, got shape {x.shape}")

  block_m = 16
  block_n = 16
  m, n = x.shape
  if m % block_m != 0 or n % block_n != 0:
    raise ValueError(
        f"shape {x.shape} must be divisible by {(block_m, block_n)}"
    )

  x_desc = jt.TensorDescriptor(
      base=x,
      shape=(m, n),
      strides=(n, 1),
      block_shape=(block_m, block_n),
  )
  out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
  grid = (m // block_m, n // block_n)
  return jt.triton_call(
      x_desc,
      kernel=tma_copy_kernel,
      out_shape=out_shape,
      grid=grid,
      num_warps=4,
      num_stages=3,
      N_COLS=n,
      BLOCK_M=block_m,
      BLOCK_N=block_n,
  )


def main() -> None:
  require_tma_capable_gpu()

  x = jnp.arange(64 * 64, dtype=jnp.float32).reshape(64, 64)

  eager_out = tma_copy(x).block_until_ready()
  np.testing.assert_allclose(eager_out, x)
  print("eager TMA copy OK")

  jit_out = jax.jit(tma_copy)(x).block_until_ready()
  np.testing.assert_allclose(jit_out, x)
  print("jit TMA copy OK")


if __name__ == "__main__":
  main()
