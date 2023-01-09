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

import functools
import os
import unittest

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
from jax import linear_util as lu
from jax import random
from jax._src import test_util as jtu
from jax._src import state
from jax._src.lax.control_flow.for_loop import for_loop
from jax.config import config
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
import jax_triton as jt
from jax_triton import pallas as pl
from jax_triton.pallas.pallas_call import _compile_jaxpr
import numpy as np
try:
  import torch
except ModuleNotFoundError:
  torch = None

config.parse_flags_with_absl()

@functools.partial(jax.jit, static_argnames=["bm", "bn", "gm", "bk",
                                             "interpret", "debug"])
def matmul(x, y, *, bm, bn, gm, bk, interpret, debug=False):
  m, n, k = x.shape[0], y.shape[1], x.shape[1]
  @functools.partial(
      pl.pallas_call, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      grid=jt.cdiv(m, bm) * jt.cdiv(n, bn))
  def matmul_kernel(x_ref, y_ref, o_ref):
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
  return matmul_kernel(x, y)

@functools.partial(jax.jit, static_argnames=["bm", "bn", "bk",
                                             "interpret", "debug"])
def matmul_block_spec(x, y, *, bm, bn, bk, interpret, debug=False):
  m, n, k = x.shape[0], y.shape[1], x.shape[1]
  @functools.partial(
      pl.pallas_call, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      in_specs=[
        pl.BlockSpec(lambda i, _: (i, 0), (bm, x.shape[1])),
        pl.BlockSpec(lambda _, j: (0, j), (y.shape[0], bn))
      ],
      out_specs=pl.BlockSpec(lambda i, j: (i, j), (bm, bn)),
      grid=(jt.cdiv(m, bm), jt.cdiv(n, bn)))
  def matmul_kernel(x_ref, y_ref, o_ref):
    acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)
    def body(i, acc_ref):
      x_block = pl.load(x_ref, (slice(None), pl.ds(i * bk, bk)))
      y_block = pl.load(y_ref, (pl.ds(i * bk, bk), slice(None)))
      acc_ref[:, :] += jnp.dot(x_block, y_block)
    acc = for_loop(k // bk, body, acc).astype(o_ref.dtype)
    o_ref[:, :] = acc
  return matmul_kernel(x, y)


class PallasTest(parameterized.TestCase):
  INTERPRET = False

  def setUp(self):
    super().setUp()
    pl.clear_caches()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)

class PallasCallTest(PallasTest):

  def test_add_one(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1.

    x = 0.
    self.assertEqual(add_one(x), 1.)

  def test_add_singleton_vector(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
        grid=1)
    def add_one(x_ref, o_ref):
      o_ref[0] = x_ref[0] + 1.

    x = jnp.array([0.], jnp.float32)
    np.testing.assert_allclose(add_one(x), jnp.array([1.], jnp.float32))

  def test_add_vector_block_spec(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), jnp.int32),
        in_specs=(pl.BlockSpec(lambda i: i, (1,)),),
        out_specs=(pl.BlockSpec(lambda i: i, (1,)),),
        grid=8, debug=False)
    def add_one(x_ref, o_ref):
      o_ref[0] = x_ref[0] + 1

    np.testing.assert_allclose(add_one(jnp.arange(8)), jnp.arange(8) + 1)

  def test_add_matrix_block_spec(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8, 8), jnp.int32),
        in_specs=(pl.BlockSpec(lambda i, j: (i, j), (2, 2)),),
        out_specs=(pl.BlockSpec(lambda i, j: (i, j), (2, 2)),),
        grid=(4, 4))
    def add_one(x_ref, o_ref):
      o_ref[:, :] = x_ref[:, :] + 1

    x = jnp.arange(64).reshape((8, 8))
    np.testing.assert_allclose(add_one(x), x + 1)

  def test_vector_indexing(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def index(x_ref, i_ref, o_ref):
      o_ref[()] = x_ref[i_ref[()]]

    x = jnp.arange(5.)
    for i in range(5):
      np.testing.assert_allclose(index(x, i), x[i])

  def test_vector_slicing(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        grid=1)
    def index(x_ref, idx_ref, o_ref):
      idx = idx_ref[()]
      o_ref[:] = x_ref[idx]

    x = jnp.arange(5.)
    for i in range(4):
      idx = jnp.arange(i, i + 2)
      np.testing.assert_allclose(index(x, idx), x[idx])

  def test_for_loop(self):
    
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def f(y_ref):
      y_ref[()] = 0.

      def body(i, _):
        y_ref[()] += 1

      for_loop(4, body, ())
    out = f()
    self.assertEqual(out, 4.)

  def test_for_loop1(self):
    
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((16, 16), jnp.float32),
        debug=True,
        grid=1)
    def f(x_ref, y_ref, o_ref):
      o_ref[:, :] = jnp.zeros((16, 16), jnp.float32)

      def body(i, _):
        o_ref[:, :]  += jnp.dot(x_ref[:, :], y_ref[:, :])

      for_loop(4, body, ())
    x = y = jnp.ones((16, 16), jnp.float32)
    out = f(x, y)
    np.testing.assert_allclose(out, 4 * x.dot(y))

  def test_for_loop2(self):
    
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((32, 32), jnp.float32),
        grid=1)
    def f(x_ref, y_ref, o_ref):
      acc = jnp.zeros((32, 32), jnp.float32)
      def body(i, acc_ref):
        acc = acc_ref[:, :]
        x = pl.load(x_ref, (pl.ds(None), pl.ds(i * 16, 16)))
        y = pl.load(y_ref, (pl.ds(i * 16, 16), pl.ds(None)))
        acc = acc + pl.dot(x, y)
        acc_ref[:, :] = acc
      acc = for_loop(2, body, acc)
      o_ref[:, :] = acc
    x = y = jnp.ones((32, 32), jnp.float32)
    out = f(x, y)
    np.testing.assert_allclose(out, x.dot(y))

  def test_for_loop3(self):
    
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((32, 32), jnp.float32),
        grid=(2, 2))
    def f(x_ref, y_ref, o_ref):
      pid_m, pid_n = pl.program_id(0), pl.program_id(1)
      acc = jnp.zeros((16, 16), jnp.float32)
      def body(i, acc_ref):
        acc = acc_ref[:, :]
        x = pl.load(x_ref, (pl.ds(pid_m * 16, 16), pl.ds(i * 16, 16)))
        y = pl.load(y_ref, (pl.ds(i * 16, 16), pl.ds(pid_n * 16, 16)))
        acc = acc + pl.dot(x, y)
        acc_ref[:, :] = acc
      acc = for_loop(2, body, acc)
      pl.store(o_ref, (pl.ds(pid_m * 16, 16), pl.ds(pid_n * 16, 16)), acc)
    x = y = jnp.ones((32, 32), jnp.float32)
    out = f(x, y)
    np.testing.assert_allclose(out, x.dot(y))

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
      if block_size_m <= m and block_size_n <= n and block_size_k <= k
    ])
  def test_matmul(self, m, n, k, dtype, bm, bn, bk, gm):
    self.skipTest("slow")

    # TODO(sharadmv): expose this information in `jaxlib`
    if torch is not None and torch.cuda.get_device_capability() < (7, 0):
      raise unittest.SkipTest(
          "Matmul only works on GPUs with capability >= sm70")

    k1, k2 = random.split(random.PRNGKey(0))
    x = random.normal(k1, (m, k), dtype=dtype)
    y = random.normal(k2, (k, n), dtype=dtype)
    out, expected = matmul(x, y, bm=bm, bn=bn, bk=bk, gm=gm,
                           interpret=self.INTERPRET), jnp.matmul(x, y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.named_parameters(*[
    (f"m_{m}_n_{n}_k_{k}_dtype_{dtype}_bm_{block_size_m}_"
     f"bn_{block_size_n}_bk_{block_size_k}", m, n, k, dtype,
     block_size_m, block_size_n, block_size_k)
      for m in [512, 1024]
      for k in [512]
      for n in [512, 1024]
      for dtype in ["float32", "float16"]
      for block_size_m in [64, 128]
      for block_size_n in [128, 256]
      for block_size_k in [32]
      if block_size_m <= m and block_size_n <= n and block_size_k <= k
    ])
  def test_matmul_block_spec(self, m, n, k, dtype, bm, bn, bk):
    self.skipTest("Oof")

    # TODO(sharadmv): expose this information in `jaxlib`
    if torch is not None and torch.cuda.get_device_capability() < (7, 0):
      raise unittest.SkipTest(
          "Matmul only works on GPUs with capability >= sm70")

    k1, k2 = random.split(random.PRNGKey(0))
    x = random.normal(k1, (m, k), dtype=dtype)
    y = random.normal(k2, (k, n), dtype=dtype)
    out, expected = matmul_block_spec(x, y, bm=bm, bn=bn, bk=bk,
                                      interpret=self.INTERPRET), jnp.matmul(x, y)
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
        self.pallas_call,
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
    @functools.partial(self.pallas_call,
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
    @functools.partial(self.pallas_call,
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
        self.pallas_call,
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
        self.pallas_call,
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
    out = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid, input_output_aliases={0: 0})(x)
    expected = x + 1
    np.testing.assert_allclose(out, expected)

  @parameterized.named_parameters(*[
    ("add_i32", pl.atomic_add, np.array(1, np.int32), np.sum),
    ("max_i32", pl.atomic_max, np.array(1, np.int32), np.max),
    ("min_i32", pl.atomic_min, np.array(1, np.int32), np.min),
    ("add_f16", pl.atomic_add, np.array(1, np.float16), np.sum),
    ("add_f32", pl.atomic_add, np.array(1, np.float32), np.sum),
    ("max_f32", pl.atomic_max, np.array(1, np.float32), np.max),
    ("min_f32", pl.atomic_min, np.array(1, np.float32), np.min),
  ])
  def test_scalar_atomic_scalar(self, op, value, numpy_op):
    # TODO(sharadmv): expose this information in `jaxlib`
    if torch is not None and torch.cuda.get_device_capability() < (7, 0):
      raise unittest.SkipTest(
          "Atomic ops only works on GPUs with capability >= sm70")

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), value.dtype),
        grid=(1,),
        debug=True,
        input_output_aliases={1: 0})
    def atomic_kernel(x_ref, _, o_ref):
      op(o_ref, (), x_ref[()])
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
          "Atomic ops only works on GPUs with capability >= sm70")

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), value.dtype),
        grid=value.shape[0],
        debug=True,
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
        self.pallas_call,
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


  @parameterized.named_parameters(*[
    (f"{op_name}_{dtype}_{axis}", op, dtype, axis)
    for op_name, op in [
      ("add", jnp.sum),
      ("max", jnp.max),
      ("min", jnp.min),
      ("argmax", jnp.argmax),
      ("argmin", jnp.argmin),
    ]
    for axis in [0, 1]
    for dtype in ["float16", "float32", "int32"]
    ])
  def test_array_reduce(self, op, dtype, axis):
    m, n = 32, 8
    out_dtype = dtype
    if op in {jnp.argmin, jnp.argmax}:
      out_dtype = jnp.int32
    out_shape = jax.ShapeDtypeStruct((n if axis == 0 else m,), out_dtype)
    @functools.partial(
        self.pallas_call,
        out_shape=out_shape,
        grid=1)
    def reduce(x_ref, y_ref):
      x = pl.load(x_ref, (jnp.arange(m), jnp.arange(n)))
      y = op(x, axis=axis)
      pl.store(y_ref, (jnp.arange(y.shape[0]),), y)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jnp.arange(m * n, dtype=dtype).reshape((m, n))
    else:
      x = random.normal(random.PRNGKey(0), (m, n), dtype=dtype)
    y = reduce(x)
    y_ref = op(x, axis=axis)
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)

  def test_using_pallas_slice(self):
    m, n = 32, 4
    out_shape = jax.ShapeDtypeStruct((4, n), jnp.float32)
    @functools.partial(
        self.pallas_call,
        out_shape=out_shape,
        grid=1)
    def slice_kernel(x_ref, y_ref):
      x = pl.load(x_ref, (pl.dslice(0, 4), pl.dslice(0, 4)))
      pl.store(y_ref, (pl.dslice(4), pl.dslice(4)), x)
    x = random.normal(random.PRNGKey(0), (m, n))
    y = slice_kernel(x)
    y_ref = x[:4]
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)

  def test_pallas_trace_cache(self):
    trace_count = 0
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def add_one(x_ref, o_ref):
      nonlocal trace_count
      o_ref[()] = x_ref[()] + 1.
      trace_count += 1

    @jax.jit
    def f(x):
      return add_one(add_one(x))

    self.assertEqual(f(0.), 2.)
    self.assertEqual(trace_count, 1)

  def test_pallas_compilation_cache(self):
    if self.INTERPRET:
      raise unittest.SkipTest("No Triton compilation in interpreter mode.")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        grid=1)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1.

    @jax.jit
    def f(x):
      return add_one(add_one(x))

    self.assertEqual(f(0.), 2.)
    num_misses = _compile_jaxpr.cache_info().misses
    self.assertEqual(num_misses, 1)

class PallasCallInterpreterTest(PallasCallTest):
  INTERPRET = True

class PallasCallAutodifferentiationTest(PallasTest):

  @parameterized.named_parameters(*[
    ("square", lambda x: x * x),
    ("add_one", lambda x: x + 1.),
    ("exp", jnp.exp),
    # ("tanh", jnp.tanh),  TODO(sharadmv): re-enable this case when libdevice is
    # updated
    ])
  def test_jvp(self, impl):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        debug=False,
        grid=1)
    def pallas_impl(x_ref, o_ref):
      x = x_ref[()]
      o_ref[()] = impl(x)

    k1, k2 = random.split(random.PRNGKey(0))
    x = random.normal(k1)
    t = random.normal(k2)
    out_primal, out_tangent = jax.jvp(pallas_impl, (x,), (t,))
    out_primal_ref, out_tangent_ref = jax.jvp(impl, (x,), (t,))
    np.testing.assert_allclose(out_primal, out_primal_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(out_tangent, out_tangent_ref, atol=1e-5,
                               rtol=1e-5)
    jtu.check_grads(pallas_impl, (x,), modes=["fwd"], order=2)

  @parameterized.named_parameters(*[
    ("square", lambda x: x * x),
    ("add_one", lambda x: x + 1.),
    ("exp", jnp.exp),
    # ("tanh", jnp.tanh),  TODO(sharadmv): re-enable this case when libdevice is
    # updated
    ])
  def test_jvp_slice(self, impl):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        debug=False,
        grid=1)
    def pallas_impl(x_ref, o_ref):
      x = x_ref[jnp.arange(2)]
      o_ref[jnp.arange(2)] = jnp.zeros(2)
      o_ref[2 + jnp.arange(2)] = impl(x)

    k1, k2 = random.split(random.PRNGKey(0))
    x = random.normal(k1, (8,))
    t = random.normal(k2, (8,))
    out_primal, out_tangent = jax.jvp(pallas_impl, (x,), (t,))
    out_primal_ref, out_tangent_ref = jax.jvp(
        lambda x: jnp.concatenate([jnp.zeros(2), impl(x[:2])]), (x,), (t,))
    np.testing.assert_allclose(out_primal, out_primal_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(out_tangent, out_tangent_ref, atol=1e-5,
                               rtol=1e-5)
    jtu.check_grads(pallas_impl, (x,), modes=["fwd"], order=2)

  # TODO(sharadmv): enable this when we update Triton
  # def test_jvp_matmul(self):
  #   k1, k2 = random.split(random.PRNGKey(0))
  #   x = random.normal(k1, (256, 128))
  #   y = random.normal(k2, (128, 64))
  #   bm, bn, bk, gm = 64, 128, 32, 8
  #   mm = functools.partial(matmul, bm=bm, bn=bn, bk=bk, gm=gm,
  #                          interpret=self.INTERPRET)
  #   jtu.check_grads(mm, (x, y), modes=["fwd"], order=1)

  def test_slicing_block_spec(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        in_specs=[
          pl.BlockSpec(lambda _: (0, 0), (None, 4)),
          pl.BlockSpec(lambda _: (1, 0), (None, 4)),
        ],
        out_specs=None,
        debug=False, grid=1)
    def add_vectors(x_ref, y_ref, o_ref):
      o_ref[:] = x_ref[:] + y_ref[:]
    xy = jnp.arange(8.).reshape((2, 4))
    out = add_vectors(xy, xy)
    out_ref = xy[0] + xy[1]
    np.testing.assert_allclose(out, out_ref)


class PallasCallVmapTest(PallasTest):

  def test_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False,
        grid=1)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(add_one)(jnp.arange(8))
    out_ref = jnp.arange(1, 9)
    np.testing.assert_allclose(out, out_ref)

  def test_double_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False,
        grid=1)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(jax.vmap(add_one))(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_slicing_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        debug=False,
        grid=(2,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    out = jax.vmap(add_one)(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_slicing_kernel_different_axes(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        debug=False,
        grid=(2,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    add_one_ref = lambda x: x + 1
    x = jnp.arange(8).reshape((2, 4))

    out = jax.vmap(add_one, in_axes=1, out_axes=1)(x)
    out_ref = jax.vmap(add_one_ref, in_axes=1, out_axes=1)(x)
    np.testing.assert_allclose(out, out_ref)

    out = jax.vmap(add_one, in_axes=1, out_axes=0)(x)
    out_ref = jax.vmap(add_one_ref, in_axes=1, out_axes=0)(x)
    np.testing.assert_allclose(out, out_ref)

  def test_double_vmap_of_slicing_kernel_different_axes(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        debug=False,
        grid=(4,))
    def sin(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = jnp.sin(x_ref[i])
    sin_ref = jnp.sin
    x = jnp.arange(64.).reshape((8, 4, 2))

    out = jax.vmap(jax.vmap(sin, in_axes=1), in_axes=0)(x)
    out_ref = jax.vmap(jax.vmap(sin_ref, in_axes=1), in_axes=0)(x)
    np.testing.assert_allclose(out, out_ref, atol=1e-3, rtol=1e-3)

class PallasCallInterpreterVmapTest(PallasCallVmapTest):
  INTERPRET = True

class PallasPrimitivesTest(parameterized.TestCase):

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)), "<- a[:,:,:]"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)), "<- a[:3,:,:]"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)), "<- a[1:4,:,:4]"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)), "<- a[b,:,:4]"),
    (lambda: (jnp.arange(5), jnp.arange(3), jnp.arange(4)), "<- a[e,f,g]"),
  ])
  def test_load_pretty_print(self, expr, expected):
    def body(x_ref):
      x = pl.load(x_ref, expr())
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)), "a[:,:,:] <-"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)), "a[:3,:,:] <-"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)), "a[1:4,:,:4] <-"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)), "a[b,:,:4] <-"),
    (lambda: (jnp.arange(5), jnp.arange(3), jnp.arange(4)), "a[l,m,n] <-"),
  ])
  def test_store_pretty_print(self, expr, expected):
    def body(x_ref):
      pl.store(x_ref, expr(), pl.load(x_ref, expr()))
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

if __name__ == "__main__":
  absltest.main()
