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
import jax.numpy as jnp
from jax import config
from jax import random
import jax_triton as jt
import numpy as np
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell as gl_blackwell


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


@gluon.jit
def tma_copy_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr):
  # Copies a single (M, N) tile from in_ptr to out_ptr entirely via TMA:
  # an async load into shared memory, gated on an mbarrier, followed by an
  # async store back out to global memory.
  layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([M, N], gl.float32)

  in_desc = gl_blackwell.tma.make_tensor_descriptor(
    in_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N], layout=layout,
  )
  out_desc = gl_blackwell.tma.make_tensor_descriptor(
    out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N], layout=layout,
  )

  smem = gl.allocate_shared_memory(gl.float32, [M, N], layout)
  bar = gl_blackwell.mbarrier.allocate_mbarrier()
  gl_blackwell.mbarrier.init(bar, count=1)

  gl_blackwell.mbarrier.expect(bar, bytes_per_cta=in_desc.nbytes_per_cta)
  gl_blackwell.tma.async_copy_global_to_shared(in_desc, [0, 0], bar, smem)
  gl_blackwell.mbarrier.wait(bar, phase=0)
  gl_blackwell.mbarrier.invalidate(bar)

  gl_blackwell.tma.async_copy_shared_to_global(out_desc, [0, 0], smem)
  gl_blackwell.tma.store_wait(0)


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
        out_shape=jax.ShapeDtypeStruct(shape=input.shape, dtype=input.dtype),
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
        out_shape=jax.ShapeDtypeStruct(shape=input.shape, dtype=input.dtype),
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
        out_shape=jax.ShapeDtypeStruct(shape=input.shape, dtype=input.dtype),
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
        out_shape=jax.ShapeDtypeStruct(shape=input.shape, dtype=input.dtype),
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
  def test_tma_copy(self, shape):
    """Round-trips an (M, N) tile through shared memory using TMA load/store.

    TODO: Add Hopper tests as well, the current test is only for Blackwell.
    Requires a Blackwell-or-newer GPU (TMA hardware) and on-device construction
    of tensor descriptors, which in turn needs a jaxlib new enough to
    allocate the scratch buffer Triton's compiler implicitly appends as a
    kernel argument (see jaxlib/gpu/triton_kernels.cc, KernelCall::Launch).
    """
    if jax.devices()[0].platform != "gpu":
      self.skipTest("TMA requires an NVIDIA GPU.")
    compute_capability = jt.get_compute_capability(0)
    if compute_capability < 100:
      self.skipTest("TMA requires Blackwell (sm_100) or newer.")

    M, N = shape
    dtype = jnp.float32

    def tma_copy(input):
      return jt.triton_call(
        input,
        kernel=tma_copy_kernel,
        out_shape=jax.ShapeDtypeStruct(shape=input.shape, dtype=input.dtype),
        grid=1,
        M=M,
        N=N,
      )

    input = random.uniform(random.key(0), (M, N), dtype=dtype)
    output = tma_copy(input)
    np.testing.assert_array_equal(output, input)


if __name__ == "__main__":
  absltest.main()
