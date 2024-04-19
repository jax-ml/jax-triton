# Copyright 2024 The jax_triton Authors.
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

"""H100 clustering tests."""

import functools
import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


def _dummy_fn(x):
  assert x.size % 4 == 0

  @triton.jit
  def dummy_kernel(x_ptr, o_ptr):
    offs = tl.program_id(axis=0) * 4 + tl.arange(0, 4)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs))

  return jt.triton_call(x, kernel=dummy_kernel, out_shape=x, grid=(x.size // 4))


class ClusterTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4, 8)
  def test_cluster(self, num_ctas):
    if 'h100' not in jax.devices()[0].device_kind.lower():
      self.skipTest('Clusters available only on H100s.')

    cluster_dims = []
    original_compile_ttir_to_ptx_fn = jt.triton_lib.compile_ttir_to_ptx_inplace

    def my_compile_ttir_to_ptx(*args, **kwargs):
      nonlocal cluster_dims, original_compile_ttir_to_ptx_fn
      ret_args = original_compile_ttir_to_ptx_fn(*args, **kwargs)
      cluster_dims = ret_args.cluster_dims
      return ret_args

    my_triton_call = functools.partial(jt.triton_call, num_ctas=num_ctas)

    with mock.patch.object(jt, 'triton_call', my_triton_call):
      with mock.patch.object(
          jt.triton_lib, 'compile_ttir_to_ptx_inplace', my_compile_ttir_to_ptx
      ):
        _dummy_fn(jnp.empty((16,)))
        self.assertEqual(math.prod(cluster_dims), num_ctas)

  def test_invalid_cluster_size(self):
    if 'h100' not in jax.devices()[0].device_kind.lower():
      self.skipTest('Clusters available on H100s.')

    my_triton_call = functools.partial(jt.triton_call, num_ctas=9)

    with mock.patch.object(jt, 'triton_call', my_triton_call):
      with self.assertRaises(Exception):
        _dummy_fn(jnp.empty((16,)))

  def test_cluster_not_available(self):
    if 'h100' in jax.devices()[0].device_kind.lower():
      self.skipTest('Clusters available only on H100s.')

    my_triton_call = functools.partial(jt.triton_call, num_ctas=2)

    with mock.patch.object(jt, 'triton_call', my_triton_call):
      with self.assertRaises(ValueError):
        _dummy_fn(jnp.empty((16,)))


if __name__ == '__main__':
  absltest.main()
