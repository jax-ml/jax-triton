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
def memcpy_kernel(
    in_ptr, out_ptr, size: gl.constexpr, block_size: gl.constexpr
):
  pid = gl.program_id(0)
  start = pid * block_size
  end = min(start + block_size, size)
  for i in range(start, end):
    value = gl.load(in_ptr + i)
    gl.store(out_ptr + i, value)


def _make_tma_copy_kernel(arch):
  @gluon.jit
  def tma_copy_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr):
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


class GluonTest(parameterized.TestCase):

  @parameterized.product(dtype=_JAX_DTYPES)
  def test_copy_scalar(self, dtype):
    x = jnp.array(42.314, dtype=dtype)
    output = jt.triton_call(
        x,
        kernel=copy_scalar_kernel,
        out_type=jax.typeof(x),
        grid=1,
    )
    np.testing.assert_equal(output, x)

  @parameterized.product(block_size=[64], size=[40, 500, 16 * 1024 + 1])
  def test_memcpy(self, block_size, size):
    x = random.uniform(random.key(0), (size,), dtype=jnp.float32)
    output = jt.triton_call(
        x,
        size=size,
        kernel=memcpy_kernel,
        out_type=jax.typeof(x),
        grid=(triton.cdiv(size, block_size),),
        num_warps=1,
        block_size=block_size,
    )
    np.testing.assert_array_equal(output, x)

  @parameterized.product(block_size=[64], size=[40, 500, 16 * 1024 + 1])
  def test_memcpy_inplace(self, block_size, size):
    def fn(x, y_ref):
      return jt.triton_call(
          x,
          y_ref,
          size,
          kernel=memcpy_kernel,
          out_type=(),
          grid=(triton.cdiv(size, block_size),),
          num_warps=1,
          block_size=block_size,
      )

    x = random.uniform(random.key(0), (size,), dtype=jnp.float32)
    y_ref = jax.new_ref(jnp.empty_like(x))
    fn(x, y_ref)
    np.testing.assert_array_equal(y_ref[...], x)

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
