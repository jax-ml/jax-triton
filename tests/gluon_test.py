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

"""Gluon-specific tests."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import config
from jax import random
from jax._src import test_util as jtu
from jax._src.lib import version as jaxlib_version
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell as gl_blackwell
from triton.experimental.gluon.language.nvidia import hopper as gl_hopper


config.parse_flags_with_absl()


def setUpModule():
  config.update("jax_enable_x64", True)


def tearDownModule():
  config.update("jax_enable_x64", False)


_JAX_DTYPES = [
  jnp.float64,
  jnp.float32,
  jnp.float16,
  jnp.int64,
  jnp.int32,
  jnp.int16,
  jnp.int8,
  jnp.uint64,
  jnp.uint32,
  jnp.uint16,
  jnp.uint8,
]


@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
  value = gl.load(in_ptr)
  gl.store(out_ptr, value)


@gluon.jit
def memcpy_kernel(in_ptr, xnumel, out_ptr, XBLOCK: gl.constexpr):
  # Each program processes the addresses [pid, pid + BLOCK_X), clamped into
  # the range [0, xnumel).
  pid = gl.program_id(0)
  start = pid * XBLOCK
  end = min(start + XBLOCK, xnumel)
  for i in range(start, end):
    value = gl.load(in_ptr + i)
    gl.store(out_ptr + i, value)


@gluon.jit
def memcpy_inplace_output_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
  # Each program processes the addresses [pid, pid + BLOCK_X), clamped into
  # the range [0, xnumel).
  pid = gl.program_id(0)
  start = pid * XBLOCK
  end = min(start + XBLOCK, xnumel)
  for i in range(start, end):
    value = gl.load(in_ptr + i)
    gl.store(out_ptr + i, value)


def _make_tma_copy_kernel(arch):
  @gluon.jit
  def tma_copy_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr):
    # Copies an (M, N) tile using on-device TMA descriptors and mbarrier.
    layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [M, N], gl.float32
    )
    in_desc = arch.tma.make_tensor_descriptor(
        in_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N], layout=layout
    )
    out_desc = arch.tma.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M, N],
        layout=layout,
    )
    smem = gl.allocate_shared_memory(gl.float32, [M, N], layout)
    bar = arch.mbarrier.allocate_mbarrier()
    arch.mbarrier.init(bar, count=1)
    arch.mbarrier.expect(bar, bytes_per_cta=in_desc.nbytes_per_cta)
    arch.tma.async_copy_global_to_shared(in_desc, [0, 0], bar, smem)
    arch.mbarrier.wait(bar, phase=0)
    arch.mbarrier.invalidate(bar)
    arch.tma.async_copy_shared_to_global(out_desc, [0, 0], smem)
    arch.tma.store_wait(0)

  return tma_copy_kernel


tma_copy_kernel_hopper = _make_tma_copy_kernel(gl_hopper)
tma_copy_kernel_blackwell = _make_tma_copy_kernel(gl_blackwell)


# autotuner example isn't ported as Triton's autotuner depends on torch internally


class GluonTest(parameterized.TestCase):
  @parameterized.product(dtype=_JAX_DTYPES)
  def test_copy_scalar_kernel(self, dtype):
    def copy_scalar(input: jnp.ndarray) -> jnp.ndarray:
      assert input.size == 1 and input.ndim == 0
      # note, this also checks behaviour in the absence of metaparams args.
      return jt.triton_call(
          input,
          kernel=copy_scalar_kernel,
          out_type=jax.typeof(input),
          grid=1,
      )

    input = jnp.array(42.314, dtype=dtype)
    output = copy_scalar(input)
    np.testing.assert_equal(output, input)

  @parameterized.product(XBLOCK=[64], xnumel=[40, 500, 16 * 1024 + 1])
  def test_memcpy(self, XBLOCK, xnumel):
    dtype = jnp.float32

    def memcpy(input, XBLOCK):
      xnumel = input.size
      return jt.triton_call(
          input,
          xnumel,
          kernel=memcpy_kernel,
          out_type=jax.typeof(input),
          grid=(triton.cdiv(xnumel, XBLOCK),),
          num_warps=1,
          XBLOCK=XBLOCK,
      )

    input = random.uniform(random.key(0), (xnumel,), dtype=dtype)
    output = memcpy(input, XBLOCK)
    np.testing.assert_array_equal(output, input)

  @parameterized.product(XBLOCK=[64], xnumel=[40, 500, 16 * 1024 + 1])
  def test_memcpy_inplace_output(self, XBLOCK, xnumel):
    """A variation of memcpy test with a pre-allocated output buffer.
    Note that the buffer is still being copied by JAX upon kernel launch, so the kernel
    could modify it."""
    dtype = jnp.float32

    def memcpy_inplace_output(input, output, XBLOCK):
      assert input.size == output.size and input.dtype == output.dtype
      assert input.shape == output.shape
      xnumel = input.size
      return jt.triton_call(
          input,
          output,
          xnumel,
          kernel=memcpy_inplace_output_kernel,
          out_type=jax.typeof(input),
          input_output_aliases={1: 0},
          grid=(triton.cdiv(xnumel, XBLOCK),),
          num_warps=1,
          XBLOCK=XBLOCK,
      )

    input = random.uniform(random.key(0), (xnumel,), dtype=dtype)
    output = jnp.empty_like(input)
    # without a buffer donation, JAX will still make a copy of output before passing
    # it to the kernel, so the kernel could modify it. Hence we have to accept that
    # inout-output buffer copy as a result here.
    result = memcpy_inplace_output(input, output, XBLOCK)
    np.testing.assert_array_equal(result, input)

  @parameterized.product(XBLOCK=[64], xnumel=[40, 500, 16 * 1024 + 1])
  def test_memcpy_inplace_output_donate(self, XBLOCK, xnumel):
    """A variation of memcpy test with pre-allocated output buffer and buffer donation,
    preventing buffer copy being made by JAX."""
    dtype = jnp.float32

    @jax.jit(static_argnums=(2,), donate_argnums=(1,))
    def memcpy_inplace_output_donate(input, output, XBLOCK):
      assert input.size == output.size and input.dtype == output.dtype
      assert input.shape == output.shape
      xnumel = input.size
      return jt.triton_call(
          input,
          output,
          xnumel,
          kernel=memcpy_inplace_output_kernel,
          out_type=jax.typeof(input),
          input_output_aliases={1: 0},
          grid=(triton.cdiv(xnumel, XBLOCK),),
          num_warps=1,
          XBLOCK=XBLOCK,
      )

    input = random.uniform(random.key(0), (xnumel,), dtype=dtype)

    output = jnp.empty_like(input)
    output_ptr = output.unsafe_buffer_pointer()

    result = memcpy_inplace_output_donate(input, output, XBLOCK)
    # we still have to use a dedicated result object, but this time it should reuse
    # the same underlying data buffer as was allocated for the output, so no additional
    # allocation/copy should happen.

    np.testing.assert_(output.is_deleted())
    np.testing.assert_equal(output_ptr, result.unsafe_buffer_pointer())
    np.testing.assert_array_equal(result, input)

  @parameterized.product(shape=[(32, 32), (8, 64), (16, 128)])
  def test_device_side_tma_copy(self, shape):
    if jaxlib_version <= (0, 11, 0):
      self.skipTest("Device-side TMA scratch requires a newer jaxlib")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("TMA requires Hopper or newer")

    kernel = (
        tma_copy_kernel_blackwell
        if jtu.is_cuda_compute_capability_at_least("10.0")
        else tma_copy_kernel_hopper
    )
    M, N = shape
    x = random.uniform(random.key(0), (M, N), dtype=jnp.float32)
    output = jt.triton_call(
        x, kernel=kernel, out_type=jax.typeof(x), grid=1, M=M, N=N
    )
    np.testing.assert_array_equal(output, x)


if __name__ == "__main__":
  absltest.main()
