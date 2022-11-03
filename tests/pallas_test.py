# Copyright 2022 The jax_triton Authors.
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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import unittest

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
from jax import random
from jax.config import config
from jax._src.lax.control_flow.for_loop import for_loop
import jax.numpy as jnp
import jax_triton as jt
from jax_triton import pallas as pl
import numpy as np
try:
  import torch
except ModuleNotFoundError:
  torch = None

config.parse_flags_with_absl()

class PallasCallTest(parameterized.TestCase):

  def test_add_one(self):
    @functools.partial(
        pl.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1.

    x = 0.
    self.assertEqual(add_one(x), 1.)

  def test_vector_indexing(self):
    @functools.partial(
        pl.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def index(x_ref, i_ref, o_ref):
      o_ref[()] = x_ref[i_ref[()]]

    x = jnp.arange(5.)
    for i in range(5):
      np.testing.assert_allclose(index(x, i), x[i])

  def test_vector_slicing(self):
    @functools.partial(
        pl.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        grid=1)
    def index(x_ref, idx_ref, o_ref):
      idx = idx_ref[()]
      o_ref[:] = x_ref[idx]

    x = jnp.arange(5.)
    for i in range(4):
      idx = jnp.arange(i, i + 2)
      np.testing.assert_allclose(index(x, idx), x[idx])


  @parameterized.named_parameters(*[
    (f"m_{m}_n_{n}_k_{k}_dtype_{dtype}_bm_{block_size_m}_"
     f"bn_{block_size_n}_bk_{block_size_k}_gm_{group_size_m}", m, n, k, dtype,
     block_size_m, block_size_n, block_size_k, group_size_m)
      for m in [512, 1024]
      for k in [512]
      for n in [512, 1024]
      for dtype in ["float32", "float16"]
      for block_size_m in [64, 128]
      for block_size_n in [128, 256]
      for block_size_k in [32]
      for group_size_m in [8]
      if block_size_m < m and block_size_n < n and block_size_k < k
    ])
  def test_matmul(self, m, n, k, dtype, bm, bn, bk, gm):

    # TODO(sharadmv): expose this information in `jaxlib`
    if torch is not None and torch.cuda.get_device_capability() < (7, 0):
      raise unittest.SkipTest(
          "Matmul only works on GPUs with capability >= sm70")

    @functools.partial(
        pl.pallas_call, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=jt.cdiv(m, bm) * jt.cdiv(n, bn))
    def matmul(x_ref, y_ref, o_ref):
      pid = pl.program_id(axis=0)
      num_pid_m = m // bm
      num_pid_n = n // bn
      num_pid_in_group = gm * num_pid_n
      group_id = lax.div(pid, num_pid_in_group)
      first_pid_m = group_id * gm
      group_size_m = jnp.minimum(num_pid_m - first_pid_m, gm)
      pid_m = first_pid_m + lax.rem(pid, group_size_m)
      pid_n = lax.div(lax.rem(pid, num_pid_in_group), group_size_m)
      idx_m = pid_m * bm + jnp.arange(bm)
      idx_n = pid_n * bn + jnp.arange(bn)
      idx_m = pl.max_contiguous(pl.multiple_of(idx_m, bm), bm)
      idx_n = pl.max_contiguous(pl.multiple_of(idx_n, bn), bn)
      acc = jnp.zeros((bm, bn), dtype=jnp.float32)
      def body(i, acc_ref):
        idx_k = i * bk + jnp.arange(bk)
        x_idx = (
            jax.lax.broadcast_in_dim(idx_m, (bm, bk), (0,)),
            jax.lax.broadcast_in_dim(idx_k, (bm, bk), (1,)))
        y_idx = (
            jax.lax.broadcast_in_dim(idx_k, (bk, bn), (0,)),
            jax.lax.broadcast_in_dim(idx_n, (bk, bn), (1,)))
        x_block, y_block = x_ref[x_idx], y_ref[y_idx]
        out = jnp.dot(x_block, y_block)
        acc_ref[:, :] += out
      acc = for_loop(k // bk, body, acc).astype(o_ref.dtype)
      o_idx = (
          jax.lax.broadcast_in_dim(idx_m, (bm, bn), (0,)),
          jax.lax.broadcast_in_dim(idx_n, (bm, bn), (1,)),
          )
      o_ref[o_idx] = acc

    k1, k2 = random.split(random.PRNGKey(0))
    x = random.normal(k1, (m, k), dtype=dtype)
    y = random.normal(k2, (k, n), dtype=dtype)
    out, expected = matmul(x, y), jnp.matmul(x, y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.named_parameters(*(
      dict(testcase_name=f"{size}_{dtype}", size=size, dtype=dtype)
      for size in [16, 32, 64]
      for dtype in ["float32", "float16"]
  ))
  def test_dot(self, size, dtype):
    # TODO(sharadmv): expose this information in `jaxlib`
    if torch is not None and torch.cuda.get_device_capability() < (7, 0):
      raise unittest.SkipTest(
          "Matmul only works on GPUs with capability >= sm70")

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((size, size), dtype),
        grid=1)
    def dot(x_ref, y_ref, o_ref):
      x = x_ref[:, :]
      y = y_ref[:, :]
      o_ref[:, :] = pl.dot(x, y)

    k1, k2 = random.split(random.PRNGKey(0))
    x = random.normal(k1, (size, size), dtype=dtype)
    y = random.normal(k2, (size, size), dtype=dtype)
    out, expected = dot(x, y), jnp.dot(x, y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.named_parameters(*(
      dict(testcase_name=f"{batch_size}_{size}_{block_size}_{dtype}",
           batch_size=batch_size, size=size, block_size=block_size, dtype=dtype)
      for batch_size in [1, 2, 4, 23]
      for size in [1, 2, 129, 255, 256]
      for block_size in [1, 2, 32, 64, 128, 256]
      for dtype in ["float32"]
      if size < block_size
  ))
  def test_softmax(self, batch_size, size, block_size, dtype):
    @functools.partial(pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((batch_size, size), dtype),
        grid=batch_size)
    def softmax(x_ref, o_ref):
      row_idx = pl.program_id(0)
      x_idx = jnp.arange(block_size)
      row_idxs = (row_idx, x_idx)
      mask = x_idx < x_ref.shape[1]
      row = pl.load(x_ref, row_idxs, mask=mask, other=-float("inf"))
      row_minus_max = row - jnp.max(row, axis=0)
      numerator = jnp.exp(row_minus_max)
      denominator = jnp.sum(numerator, axis=0)
      softmax_output = numerator / denominator
      pl.store(o_ref, row_idxs, softmax_output, mask=mask)

    key = random.PRNGKey(0)
    x = random.normal(key, [batch_size, size], dtype=dtype)
    np.testing.assert_allclose(softmax(x), jax.nn.softmax(x, axis=-1),
        atol=1e-5, rtol=1e-5)

  @parameterized.parameters(*(
      (size, block_size)
      for size in [1, 2, 64, 129, 1021]
      for block_size in [1, 2, 32, 64, 128]
  ))
  def test_masked_load_store(self, size, block_size):
    @functools.partial(pl.pallas_call,
        out_shape=(
          jax.ShapeDtypeStruct((size,), jnp.float32)
          ),
        grid=jt.cdiv(size, block_size))
    def add_one(x_ref, o_ref):
      idx = pl.program_id(0) * block_size + jnp.arange(block_size)
      mask = idx < x_ref.shape[0]
      x = pl.load(x_ref, (idx,), mask=mask)
      pl.store(o_ref, (idx,), x + 1., mask=mask)

    key = random.PRNGKey(0)
    x = random.normal(key, (size,))
    np.testing.assert_allclose(add_one(x), x + 1., atol=1e-5, rtol=1e-5)

  def test_broadcasted_load_store(self):
    m, n = 16, 32
    @functools.partial(
        pl.pallas_call,
        out_shape=(
          jax.ShapeDtypeStruct((m, n), jnp.float32)
          ), grid=1)
    def load(x_ref, o_ref):
      x = pl.load(x_ref, (jnp.arange(m), jnp.arange(n)))
      pl.store(o_ref, (jnp.arange(m), jnp.arange(n)), x + 1.)

    key = random.PRNGKey(0)
    x = random.normal(key, (m, n))
    np.testing.assert_allclose(load(x), x + 1., atol=1e-5, rtol=1e-5)

  def test_unused_ref(self):
    m, n = 16, 32
    @functools.partial(
        pl.pallas_call,
        out_shape=(
          jax.ShapeDtypeStruct((m, n), jnp.float32)
          ), grid=1)
    def dummy(_, o_ref):
      pl.store(o_ref, (jnp.arange(m), jnp.arange(n)), jnp.ones_like(o_ref))

    key = random.PRNGKey(0)
    x = random.normal(key, (m, n))
    np.testing.assert_allclose(dummy(x), jnp.ones_like(x), atol=1e-5, rtol=1e-5)

  def test_pallas_call_with_input_output_aliasing(self):

    def add_inplace_kernel(_, o_ref, *, block_size):
      pid = pl.program_id(axis=0)  # we use a 1d launch grid so axis is 0
      block_start = pid * block_size
      offsets = block_start + jnp.arange(block_size)
      mask = offsets < o_ref.shape[0]
      x = pl.load(o_ref, (offsets,), mask=mask)
      output = x + 1
      pl.store(o_ref, (offsets,), output, mask=mask)

    grid = (8,)
    size = 8
    dtype = "float32"
    k1 = random.PRNGKey(0)
    block_size = 1
    x = random.normal(k1, [size], dtype=dtype)
    kernel = functools.partial(add_inplace_kernel, block_size=block_size)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid, input_output_aliases={0: 0})(x)
    expected = x + 1
    np.testing.assert_allclose(out, expected)

  @parameterized.named_parameters(*[
    ("add_i32", pl.atomic_add, np.array([1, 2, 3, 4], np.int32), np.sum),
    ("max_i32", pl.atomic_max, np.array([1, 2, 3, 4], np.int32), np.max),
    ("min_i32", pl.atomic_min, np.array([1, 2, 3, 4], np.int32), np.min),
    ("add_f16", pl.atomic_add, np.array([1, 2, 3, 4], np.float16), np.sum),
    ("add_f32", pl.atomic_add, np.array([1, 2, 3, 4], np.float32), np.sum),
    ("max_f32", pl.atomic_max, np.array([1, 2, 3, 4], np.float32), np.max),
    ("min_f32", pl.atomic_min, np.array([1, 2, 3, 4], np.float32), np.min),
  ])
  def test_scalar_atomic(self, op, value, numpy_op):
    # TODO(sharadmv): expose this information in `jaxlib`
    if torch is not None and torch.cuda.get_device_capability() < (7, 0):
      raise unittest.SkipTest(
          "Atomic ops onl works on GPUs with capability >= sm70")

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), value.dtype),
        grid=value.shape[0],
        input_output_aliases={1: 0})
    def atomic_kernel(x_ref, _, o_ref):
      pid = pl.program_id(axis=0)
      op(o_ref, (), x_ref[pid])
    if op == pl.atomic_add:
      neutral = np.array(0, dtype=value.dtype)
    elif op == pl.atomic_max:
      if np.issubdtype(value.dtype, np.integer):
        neutral = np.array(np.iinfo(value.dtype).min, value.dtype)
      else:
        neutral = np.array(-float('inf'), value.dtype)
    elif op == pl.atomic_min:
      if np.issubdtype(value.dtype, np.integer):
        neutral = np.array(np.iinfo(value.dtype).max, value.dtype)
      else:
        neutral = np.array(float('inf'), value.dtype)
    elif op == pl.atomic_or:
      neutral = np.array(False, value.dtype)
    else:
      raise NotImplementedError()
    out = atomic_kernel(value, neutral)
    np.testing.assert_allclose(out, numpy_op(value))

  @parameterized.parameters(*[(0,), (1,)])
  def test_array_atomic_add(self, axis):
    # TODO(sharadmv): expose this information in `jaxlib`
    if torch is not None and torch.cuda.get_device_capability() < (7, 0):
      raise unittest.SkipTest(
          "Atomic ops onl works on GPUs with capability >= sm70")

    m, n = 32, 8
    out_shape = jax.ShapeDtypeStruct((n if axis == 0 else m,), jnp.float32)
    @functools.partial(
        pl.pallas_call,
        out_shape=out_shape,
        grid=1,
        input_output_aliases={1: 0})
    def reduce(x_ref, _, y_ref):
      x = pl.load(x_ref, (jnp.arange(m), jnp.arange(n)))
      y = jnp.sum(x, axis=axis)
      pl.atomic_add(y_ref, (jnp.arange(y.shape[0]),), y)
    x = random.normal(random.PRNGKey(0), (m, n))
    y = jnp.zeros(out_shape.shape, out_shape.dtype)
    y = reduce(x, y)
    y_ref = np.sum(x, axis=axis)
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
