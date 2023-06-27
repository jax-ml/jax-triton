# Copyright 2023 The jax_triton Authors.
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
import itertools
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
from jax_triton.pallas.pallas_call import _initial_style_open_jaxpr
from jax_triton.pallas.ops import attention
from jax_triton.pallas.ops import layer_norm
from jax_triton.pallas.ops import rms_norm
from jax_triton.pallas.ops import softmax
try:
  from jax_triton.pallas.triton_ir_lowering import compile_jaxpr
except ModuleNotFoundError:
  compile_jaxpr = None
import numpy as np


# TODO(sharadmv): Update signatures of pallas_call to correct inputs/outputs.
# pylint: disable=no-value-for-parameter


config.update("jax_traceback_filtering", "off")
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
    if compile_jaxpr:
      compile_jaxpr.cache_clear()
    _initial_style_open_jaxpr.cache_clear()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)

class PallasCallTest(PallasTest):

  def test_add_one(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32))
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

  @parameterized.parameters(*[
    ((), (2,), ()),
    ((1,), (2,), (0,)),
    ((1, 1), (2, 2), (0, 1)),
    ((), (2, 2), ()),
  ])
  def test_broadcast_in_dim(self, in_shape, out_shape, dims):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1)
    def f(x_ref, o_ref):
      x = x_ref[...]
      o_ref[...] = jax.lax.broadcast_in_dim(x, out_shape, dims)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = jax.lax.broadcast_in_dim(x, out_shape, dims)
    np.testing.assert_allclose(f(x), expected)

  @parameterized.parameters(*[
    ((2, 4), (8,)),
    ((2, 4), (8, 1)),
    ((2, 4), (1, 8)),
    ((64,), (32, 2)),
  ])
  def test_reshape(self, in_shape, out_shape):
    # TODO(sharadmv): re-enable when `reshape` works again
    self.skipTest("Reshape not yet supported in Triton-MLIR")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1)
    def f(x_ref, o_ref):
      o_ref[...] = x_ref[...].reshape(out_shape)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = x.reshape(out_shape)
    np.testing.assert_allclose(f(x), expected)

  @parameterized.parameters(*[
    ((), (1,)),
    ((), (1, 1)),
    ((2, 4), (2, 4)),
    ((2, 4), (2, 4, 1)),
    ((2, 4, 1), (2, 4)),
    ((2, 4), (1, 2, 4)),
    ((1, 2, 4), (2, 4)),
    ((2, 4), (2, 1, 4)),
    ((1, 2, 1, 4, 1), (2, 4)),
    ((2, 4,), (1, 2, 1, 4)),
    ((2, 4,), (1, 2, 4, 1)),
    ((1, 2, 4, 1), (1, 2, 1, 4, 1)),
  ])
  def test_reshape_noop_or_singleton_dims(self, in_shape, out_shape):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1)
    def f(x_ref, o_ref):
      o_ref[...] = x_ref[...].reshape(out_shape)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = x.reshape(out_shape)
    np.testing.assert_allclose(f(x), expected)

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
    if jt.get_compute_capability(0) < 70:
      raise unittest.SkipTest(
          "Matmul only works on GPUs with capability >= sm70")
    if (jt.get_compute_capability(0) <= 75
        and (bm > 128 or bn > 128 or bk > 32)):
      raise unittest.SkipTest("Block sizes too big for sm70.")
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
    if jt.get_compute_capability(0) < 70:
      raise unittest.SkipTest(
          "Matmul only works on GPUs with capability >= sm70")
    if (jt.get_compute_capability(0) <= 75
        and (bm > 128 or bn > 128 or bk > 32)):
      raise unittest.SkipTest("Block sizes too big for sm70.")

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
    if jt.get_compute_capability(0) < 70:
      raise unittest.SkipTest(
          "Matmul only works on GPUs with capability >= sm70")

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((size, size), dtype),
        grid=1)
    def dot(x_ref, y_ref, o_ref):
      x = x_ref[:, :]
      y = y_ref[:, :]
      o_ref[:, :] = pl.dot(x, y).astype(o_ref.dtype)

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
    ("add_i32", pl.atomic_add, np.array([1, 2, 3, 4], np.int32), np.sum),
    ("max_i32", pl.atomic_max, np.array([1, 2, 3, 4], np.int32), np.max),
    ("min_i32", pl.atomic_min, np.array([1, 2, 3, 4], np.int32), np.min),
    ("add_f16", pl.atomic_add, np.array([1, 2, 3, 4], np.float16), np.sum),
    ("add_f32", pl.atomic_add, np.array([1, 2, 3, 4], np.float32), np.sum),
    ("max_f32", pl.atomic_max, np.array([1, 2, 3, 4], np.float32), np.max),
    ("min_f32", pl.atomic_min, np.array([1, 2, 3, 4], np.float32), np.min),
  ])
  def test_scalar_atomic(self, op, value, numpy_op):
    if jt.get_compute_capability(0) < 70:
      raise unittest.SkipTest(
          "Atomic ops onl works on GPUs with capability >= sm70")

    @functools.partial(
        self.pallas_call,
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
    if jt.get_compute_capability(0) < 70:
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

  @parameterized.parameters(False, True)
  def test_reduce_only_dim(self, use_store):
    m = 32
    x = random.normal(random.PRNGKey(0), (m,), dtype=jnp.float32)
    out_shape = jax.ShapeDtypeStruct((), x.dtype)
    @functools.partial(
        self.pallas_call,
        out_shape=out_shape,
        grid=1, debug=True)
    def reduce(x_ref, y_ref):
      x = pl.load(x_ref, (jnp.arange(m),))
      y = jnp.sum(x, axis=-1)
      if use_store:
        pl.store(y_ref, (), y)
      else:
        y_ref[...] = y
    y = reduce(x)
    y_ref = jnp.sum(x, axis=-1)
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
    for axis in [0, 1, (1,), (0, 1)]
    for dtype in ["float16", "float32", "int32", "uint32"]
    if isinstance(axis, int) or "arg" not in op_name
    ])
  def test_array_reduce(self, op, dtype, axis):
    m, n = 32, 8
    out_dtype = dtype
    if op in {jnp.argmin, jnp.argmax}:
      out_dtype = jnp.int32
    def make_x(key):
      if jnp.issubdtype(dtype, jnp.integer):
        return random.shuffle(key, jnp.arange(m * n, dtype=dtype)).reshape(m, n)
      else:
        return random.normal(key, (m, n), dtype=dtype)
    out_shape = jax.ShapeDtypeStruct(
        op(make_x(random.PRNGKey(0)), axis=axis).shape, out_dtype)
    @functools.partial(
        self.pallas_call,
        out_shape=out_shape,
        grid=1, debug=not isinstance(axis, int))
    def reduce(x_ref, y_ref):
      x = pl.load(x_ref, (jnp.arange(m), jnp.arange(n)))
      y = op(x, axis=axis)
      pl.store(y_ref, tuple(jnp.arange(d) for d in y.shape), y)
    for i, key in enumerate(random.split(random.PRNGKey(0), 20)):
      x = make_x(key)
      y = reduce(x)
      y_ref = op(x, axis=axis)
      np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2, err_msg=i)

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
    if not compile_jaxpr:
      self.skipTest("No Triton GPU.")
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
    num_misses = compile_jaxpr.cache_info().misses
    self.assertEqual(num_misses, 1)

  @parameterized.parameters(*[
    (0, 0, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1),
    (2, 1, 1),
    (2, 1, 1),
  ])
  def test_atomic_cas(self, init_value, cmp, new_value):

    @functools.partial(
        self.pallas_call, out_shape=(
          jax.ShapeDtypeStruct((), jnp.int32),
          jax.ShapeDtypeStruct((), jnp.int32)),
        input_output_aliases={0: 0})
    def swap(_, lock_ref, out_ref):
      out_ref[()] = pl.atomic_cas(lock_ref, cmp, new_value)

    lock, out = swap(init_value)
    np.testing.assert_allclose(lock, new_value if cmp == init_value else
                               init_value)
    np.testing.assert_allclose(out, init_value)

  @parameterized.parameters(*[
    1, 2, 3, 4, 8
  ])
  def test_atomic_counter(self, num_threads):
    if self.INTERPRET:
      self.skipTest("While loop not supported in interpret mode yet.")

    @functools.partial(
        self.pallas_call, out_shape=(
          jax.ShapeDtypeStruct((), jnp.int32),
          jax.ShapeDtypeStruct((), jnp.int32)),
        input_output_aliases={0: 0, 1: 1},
        grid=(num_threads,))
    def increment(_, __, lock_ref, counter_ref):
      def _cond(_):
        return pl.atomic_cas(lock_ref, 0, 1) == 1
      lax.while_loop(_cond, lambda a: a, 0)
      counter_ref[...] += 1
      pl.atomic_xchg(lock_ref, (), 0)

    lock, count = increment(0, 0)
    np.testing.assert_allclose(lock, 0)
    np.testing.assert_allclose(count, num_threads)

class PallasCallInterpreterTest(PallasCallTest):
  INTERPRET = True

class PallasControlFlowTest(PallasTest):

  def setUp(self):
    super().setUp()
    if self.INTERPRET:
      self.skipTest("Control flow not supported in interpreter mode yet.")

  def test_cond_simple(self):
    arg = jnp.float32(0.)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       debug=False)
    def f(branch_ref, x_ref, y_ref):
      y_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x: x**2, lambda x: -x),
          x_ref[...])
    y = f(jnp.int32(0), arg + 3.)
    self.assertEqual(y, 9.)
    y = f(jnp.int32(1), arg + 2.)
    self.assertEqual(y, -2.)

  def test_cond_threebranch(self):
    arg = jnp.float32(0.)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       grid=1,
                       debug=False)
    def f(branch_ref, x_ref, y_ref):
      y_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x: x**2, lambda x: -x, lambda x: -x**2),
          x_ref[...])
    y = f(jnp.int32(0), arg + 3.)
    self.assertEqual(y, 9.)
    y = f(jnp.int32(1), arg + 2.)
    self.assertEqual(y, -2.)
    y = f(jnp.int32(2), arg + 4.)
    self.assertEqual(y, -16.)

  @parameterized.parameters(1, 2, 4, 8)
  def test_cond_vectors(self, block_size):
    arg = jnp.float32([0.] * 8)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       in_specs=[pl.BlockSpec(lambda _: (), ()),
                                 pl.BlockSpec(lambda i: i, (block_size,))],
                       out_specs=pl.BlockSpec(lambda i: i, (block_size,)),
                       grid=jt.cdiv(arg.shape[0], block_size),
                       debug=False)
    def f(branch_ref, x_ref, y_ref):
      y_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x: x**2, lambda x: -x),
          x_ref[...])
    y = f(jnp.int32(0), arg + 3.)
    np.testing.assert_allclose(y, arg + 9.)
    y = f(jnp.int32(1), arg + 2.)
    np.testing.assert_allclose(y, arg - 2.)

  @parameterized.parameters(1, 2, 4, 8)
  def test_cond_threebranch_vectors(self, block_size):
    arg = jnp.float32([0.] * 8)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       in_specs=[pl.BlockSpec(lambda _: (), ()),
                                 pl.BlockSpec(lambda i: i, (block_size,))],
                       out_specs=pl.BlockSpec(lambda i: i, (block_size,)),
                       grid=jt.cdiv(arg.shape[0], block_size),
                       debug=False)
    def f(branch_ref, x_ref, y_ref):
      y_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x: x**2, lambda x: -x, lambda x: -x**2),
          x_ref[...])
    y = f(jnp.int32(0), arg + 3.)
    np.testing.assert_allclose(y, arg + 9.)
    y = f(jnp.int32(1), arg + 2.)
    np.testing.assert_allclose(y, arg - 2.)
    y = f(jnp.int32(2), arg + 4.)
    np.testing.assert_allclose(y, arg - 16.)

  @parameterized.parameters(*itertools.product([1, 8], [1, 2, 4]))
  def test_cond_threebranch_matrix_out(self, bx, by):
    x = jnp.arange(64.)[:, None]
    y = jnp.arange(128.)[None, :]
    # TODO(sharadmv): Renaming in_specs->in_spec silently breaks.
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), jnp.float32),
        in_specs=[
            pl.BlockSpec(lambda _, __: (), ()),
            pl.BlockSpec(lambda i, _: (i, 0), (bx, 1)),
            pl.BlockSpec(lambda _, j: (0, j), (1, by))],
        out_specs=pl.BlockSpec(lambda i, j: (i, j), (bx, by)),
        grid=(jt.cdiv(x.shape[0], bx), jt.cdiv(y.shape[1], by)),
        debug=False)
    def f(branch_ref, x_ref, y_ref, o_ref):
      o_ref[...] = lax.switch(
          branch_ref[...],
          (lambda x, y: (x - y)**2,
           lambda x, y: -jnp.abs(x - y),
           lambda x, y: jnp.sqrt(jnp.abs(x - y))),
          x_ref[...],
          y_ref[...])
    np.testing.assert_allclose(f(jnp.int32(0), x, y), (x - y)**2)
    np.testing.assert_allclose(f(jnp.int32(1), x, y), -jnp.abs(x - y))
    np.testing.assert_allclose(f(jnp.int32(2), x, y), jnp.sqrt(jnp.abs(x - y)))

  def test_conditional_write(self):
    arg = jnp.arange(8, dtype=jnp.float32)
    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct(arg.shape, jnp.float32),
                       debug=False)
    def f(branch_ref, x_ref, out_ref):
      out_ref[...] = -x_ref[...]
      def if_true(z):
        out_ref[4] = z
        return ()
      jax.lax.cond(branch_ref[...], if_true, lambda z: (), x_ref[6])
    np.testing.assert_allclose(f(jnp.bool_(True), arg),
                               jnp.float32([0., -1, -2, -3, 6, -5, -6, -7]))
    np.testing.assert_allclose(f(jnp.bool_(False), arg),
                               -arg)

    # We actually expect the assertion failure in linearize, but this also covers another case where an effect was causing an earlier assertion failure.
    with self.assertRaises(AssertionError):
      # Notably, we should not have a ValueError for mismatched Read<N> effect.
      dx = jax.grad(lambda x: jnp.sum(f(jnp.bool_(True), x)**2))(arg)
      # np.testing.assert_allclose(
      #     dx, jnp.float32([0., 2, 4, 6, 0, 10, 12 + 12, 14]))

  def test_scan_cond_vm_explicit_ref_arg(self):
    program = jnp.int32([0, 1, 2, 3, 2])
    params = jnp.arange(len(program) * 3.).reshape(len(program), 3)
    x = jnp.arange(7.)
    bx = 4

    @jax.jit
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
        in_specs=[
            pl.BlockSpec(lambda _: (0,), program.shape),  # program
            pl.BlockSpec(lambda _: (0, 0), params.shape),  # params
            pl.BlockSpec(lambda i: (i,), (bx,))],  # x
        out_specs=pl.BlockSpec(lambda i: (i,), (bx,)),
        grid=jt.cdiv(x.shape[0], bx),
        debug=True)
    def f(program_ref, params_ref, x_ref, out_ref):
      x = x_ref[...]

      def body_fn(i, args):
        state, program_ref, params_ref = args
        opcode = program_ref[i]
        state = jax.lax.switch(
            opcode,
            (lambda state, params, i: state + params[i, 0] * 2.**i * x,
             lambda state, params, i: state + params[i, 1] * 2.**i * x,
             lambda state, params, i: state + params[i, 2] * 2.**i * x,
             lambda state, params, i: state + params[i, 1] * 2.**i * x,
             ),
            state, params_ref, i)
        return state, program_ref, params_ref
      out_ref[...] = jax.lax.fori_loop(
          0, len(program), body_fn,
          (jnp.zeros(x.shape), program_ref, params_ref))[0]

    expected = (x * params[0, 0] +
                2 * x * params[1, 1] +
                4 * x * params[2, 2] +
                8 * x * params[3, 1] +
                16 * x * params[4, 2])
    np.testing.assert_allclose(f(program, params, x), expected)

    with self.assertRaises(AssertionError):
      jax.value_and_grad(lambda params, x: f(program, params, x).sum())(
          params, x)

  def test_scan_cond_vm_closing_over_ref(self):
    # ** Difference is the closure over params_ref in the switch branches. **
    program = jnp.int32([0, 1, 2, 3, 2, -1])
    params = jnp.arange(len(program) * 3.).reshape(len(program), 3)
    x = jnp.arange(7.)
    bx = 4

    @jax.jit
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
        in_specs=[
            pl.BlockSpec(lambda _: (0,), program.shape),  # program
            pl.BlockSpec(lambda _: (0, 0), params.shape),  # params
            pl.BlockSpec(lambda i: (i,), (bx,))],  # x
        out_specs=pl.BlockSpec(lambda i: (i,), (bx,)),
        grid=jt.cdiv(x.shape[0], bx),
        debug=False)
    def f(program_ref, params_ref, x_ref, out_ref):
      x = x_ref[...]

      def body_fn(i, args):
        state, program_ref, params_ref = args
        opcode = program_ref[i] + 1
        state = jax.lax.switch(
            opcode,
            (lambda state, *_: state,
             lambda state, i: state + params_ref[i, 0] * 2.**i * x,
             lambda state, i: state + params_ref[i, 1] * 2.**i * x,
             lambda state, i: state + params_ref[i, 2] * 2.**i * x,
             lambda state, i: state + params_ref[i, 1] * 2.**i * x,
             ),
            state, i)
        return state, program_ref, params_ref
      out_ref[...] = jax.lax.fori_loop(
          0, len(program), body_fn,
          (jnp.zeros(x.shape), program_ref, params_ref))[0]

    expected = (x * params[0, 0] +
                2 * x * params[1, 1] +
                4 * x * params[2, 2] +
                8 * x * params[3, 1] +
                16 * x * params[4, 2])
    np.testing.assert_allclose(f(program, params, x), expected)

    with self.assertRaises(AssertionError):
      jax.value_and_grad(lambda params, x: f(program, params, x).sum())(
          params, x)

  def test_fori_loop_simple(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      y_ref[...] = x_ref[...]
      def body(i, _):
        y_ref[...] += 1
      lax.fori_loop(0, 5, body, None)
    y = f(0)
    self.assertEqual(y, 5)

  def test_fori_loop_with_nonzero_lower_bound(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      y_ref[...] = x_ref[...]
      def body(i, _):
        y_ref[...] += i
      lax.fori_loop(2, 5, body, None)
    y = f(6)
    self.assertEqual(y, 6 + 2 + 3 + 4)

  def test_fori_loop_accumulates(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      def body(i, acc):
        return acc + 1
      acc = lax.fori_loop(0, 5, body, 0)
      y_ref[...] = acc
    y = f(0)
    self.assertEqual(y, 5)

  def test_fori_loop_accumulates_with_index(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      def body(i, acc):
        return acc + i
      acc = lax.fori_loop(0, 5, body, 0)
      y_ref[...] = acc
    y = f(0)
    self.assertEqual(y, 10)

  def test_fori_loop_with_writing_to_index(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((8,), jnp.int32))
    def f(y_ref):
      def body(i, _):
        y_ref[i] = i
      lax.fori_loop(0, y_ref.shape[0], body, None)
    y = f()
    np.testing.assert_allclose(y, jnp.arange(8))

  def test_fori_loop_with_dynamic_indices(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(lb_ref, ub_ref, y_ref):
      y_ref[...] = 0
      def body(i, a):
        y_ref[...] += i
        return a
      lax.fori_loop(lb_ref[...], ub_ref[...], body, 1)
    y = f(2, 5)
    np.testing.assert_allclose(y, 2 + 3 + 4)
    y = f(1, 8)
    np.testing.assert_allclose(y, sum(range(1, 8)))

  def test_simple_while(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(x_ref, y_ref):
      x = x_ref[...]
      y_ref[...] = 0
      def cond(x):
        return x < 5
      def body(x):
        y_ref[...] += 1
        return x + 1
      lax.while_loop(cond, body, x)
    y = f(0)
    self.assertEqual(y, 5)

  def test_simple_while_with_only_values(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(y_ref):
      def cond(acc):
        return acc < 5
      def body(acc):
        acc += 1
        return acc
      acc = lax.while_loop(cond, body, 0)
      y_ref[...] = acc
    y = f()
    self.assertEqual(y, 5)

  def test_while_with_dynamic_condition(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(i_ref, y_ref):
      y_ref[...] = 0
      n_iter = i_ref[...]
      def cond(i):
        return i < n_iter
      def body(i):
        y_ref[...] += 1
        return i + 1
      _ = lax.while_loop(cond, body, 0)

    self.assertEqual(f(1), 1)
    self.assertEqual(f(4), 4)
    self.assertEqual(f(100), 100)

  def test_vmap_of_while_with_dynamic_condition(self):

    @functools.partial(self.pallas_call,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def f(i_ref, y_ref):
      y_ref[...] = 0
      n_iter = i_ref[...]
      def cond(i):
        return i < n_iter
      def body(i):
        y_ref[...] += 1
        return i + 1
      _ = lax.while_loop(cond, body, 0)

    x = jnp.array([1, 4, 100])
    np.testing.assert_array_equal(jax.vmap(f)(x), x)

class PallasControlFlowInterpreterTest(PallasControlFlowTest):
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
        debug=False)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(add_one)(jnp.arange(8))
    out_ref = jnp.arange(1, 9)
    np.testing.assert_allclose(out, out_ref)

  def test_vmap_of_simple_kernel_with_in_axes_None(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False)
    def add(x_ref, y_ref, o_ref):
      o_ref[()] = x_ref[()] + y_ref[()]
    out = jax.vmap(add, in_axes=(0, None))(jnp.arange(8), 1)
    out_ref = jnp.arange(1, 9)
    np.testing.assert_allclose(out, out_ref)

  def test_double_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(jax.vmap(add_one))(jnp.arange(8).reshape((4, 2)))
    out_ref = jnp.arange(1, 9).reshape((4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_quadruple_vmap_of_simple_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False)
    def add_one(x_ref, o_ref):
      o_ref[()] = x_ref[()] + 1
    out = jax.vmap(jax.vmap(jax.vmap(jax.vmap(add_one))))(
        jnp.arange(15 * 8).reshape((5, 3, 4, 2)))
    out_ref = jnp.arange(1, 15 * 8 + 1).reshape((5, 3, 4, 2))
    np.testing.assert_allclose(out, out_ref)

  def test_quadruple_vmap_of_batched_kernel(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((7,), jnp.int32),
        debug=False,
        grid=(7,))
    def add_one(x_ref, o_ref):
      i = pl.program_id(0)
      o_ref[i] = x_ref[i] + 1
    out = jax.vmap(jax.vmap(jax.vmap(jax.vmap(add_one))))(
        jnp.arange(15 * 8 * 7).reshape((5, 3, 4, 2, 7)))
    out_ref = jnp.arange(1, 15 * 8 * 7 + 1).reshape((5, 3, 4, 2, 7))
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

  def test_vmap_of_kernel_with_input_output_aliases(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        debug=False,
        input_output_aliases={1:0},
        grid=())
    def add(x_ref, _, o_ref):
      o_ref[()] = x_ref[()] + o_ref[()] + 1
    out = jax.vmap(add, in_axes=(0, None))(jnp.arange(8), 1)
    out_ref = jnp.arange(2, 10)
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

class PallasOpsTest(PallasTest):

  def test_ne(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), jnp.bool_),
        grid=1)
    def ne(x_ref, y_ref, o_ref):
      o_ref[:] = x_ref[...] != y_ref[...]

    x = jnp.ones(8)
    y = jnp.arange(8)
    not_equal = ne(x, y)
    np.testing.assert_allclose(not_equal, x != y)

  def test_isnan(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), jnp.bool_),
        grid=1)
    def isnan(x_ref, o_ref):
      o_ref[:] = jnp.isnan(x_ref[...])

    x = jnp.arange(8.)
    x = x.at[3].set(jnp.nan)
    np.testing.assert_allclose(isnan(x), jnp.isnan(x))

class PallasOpsInterpretTest(PallasOpsTest):
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
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
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
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)),
     "c:i32[4,3,2], a[:,:,:] <-"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)),
     "c:i32[3,3,2], a[:3,:,:] <-"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)),
     "c:i32[3,3,4], a[1:4,:,:4] <-"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)),
     "e:i32[5,3,4], a[b,:,:4] <-"),
    (lambda: (jnp.arange(5), jnp.arange(3), jnp.arange(4)),
     "o:i32[5,3,4], a[l,m,n] <-"),
  ])
  def test_swap_pretty_print(self, expr, expected):
    def body(x_ref):
      x = pl.swap(x_ref, expr(), pl.load(x_ref, expr()))
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

class FusedAttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(*[
      (f"{batch_size=}_{seq_len=}_{num_heads=}_{head_dim=}_{causal=}_{use_fwd=}",
       batch_size, seq_len, num_heads, head_dim, causal, use_fwd)
      for batch_size, seq_len, num_heads, head_dim, causal, use_fwd in [
          (1, 384, 1, 64, False, False),
          (2, 384, 2, 64, False, False),
          (1, 384, 1, 64, True, False),
          (2, 384, 2, 64, True, False),
          (1, 384, 8, 64, True, True),
          (2, 384, 8, 64, True, True),
          (2, 2048, 32, 64, False, False),
      ]
  ])
  def test_fused_attention_fwd(self, batch_size, seq_len, num_heads, head_dim,
                               causal, use_fwd):
    if jt.get_compute_capability(0) < 80:
      raise unittest.SkipTest(
          "Fused attention only works on GPUs with capability >= sm80")

    k1, k2, k3 = random.split(random.PRNGKey(0), 3)
    q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
    k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
    v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)

    if use_fwd:
      @jax.jit
      def impl(q, k, v):
        v, _ = jax.vjp(functools.partial(attention.mha, causal=causal), q, k, v)
        return v
    else:
      impl = functools.partial(attention.mha, causal=causal)
    o = impl(q, k, v)
    o_ref = attention.mha_reference(q, k, v, causal=causal)
    np.testing.assert_allclose(o, o_ref, atol=0.05)

  @parameterized.named_parameters(*[
      (f"{batch_size=}_{seq_len=}_{num_heads=}_{head_dim=}_{causal=}_{use_fwd=}",
       batch_size, seq_len, num_heads, head_dim, causal, use_fwd)
      for batch_size, seq_len, num_heads, head_dim, causal, use_fwd in [
          (1, 384, 1, 64, False, False),
          (2, 384, 2, 64, False, False),
          (1, 384, 1, 64, True, False),
          (2, 384, 2, 64, True, False),
          (1, 384, 8, 64, True, True),
          (2, 384, 8, 64, True, True),
          (2, 2048, 32, 64, False, False),
          (2, 2048, 32, 64, False, True),
          (4, 1024, 72, 64, False, False),
          (4, 1024, 72, 64, False, True),
      ]
  ])
  def test_fused_attention_fwd_bias(self, batch_size, seq_len, num_heads, head_dim,
                               causal, use_fwd):
    if jt.get_compute_capability(0) < 80:
      raise unittest.SkipTest(
          "Fused attention only works on GPUs with capability >= sm80")

    k1, k2, k3, k4 = random.split(random.PRNGKey(0), 4)
    q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
    k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
    v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
    bias = random.normal(k4, (batch_size, num_heads, seq_len, seq_len), dtype=jnp.float16)

    if use_fwd:
      @jax.jit
      def impl(q, k, v, bias):
        v, _ = jax.vjp(functools.partial(attention.mha, causal=causal), q, k, v, bias)
        return v
    else:
      impl = functools.partial(attention.mha, causal=causal)
    o = impl(q, k, v, bias)
    o_ref = attention.mha_reference(q, k, v, bias, causal=causal, has_bias=True)
    np.testing.assert_allclose(o, o_ref, atol=0.05)

  @parameterized.named_parameters(*[
      (f"{batch_size=}_{seq_len=}_{num_heads=}_{head_dim=}_{causal=}",
       batch_size, seq_len, num_heads, head_dim, causal)
      for batch_size, seq_len, num_heads, head_dim, causal in [
          (1, 384, 1, 32, False),
          (2, 384, 2, 32, False),
          # TODO(b/283035396): (1, 384, 1, 32, True),
          # TODO(b/283035396): (2, 384, 2, 32, True),
      ]
  ])
  def test_fused_attention_bwd(self, batch_size, seq_len, num_heads, head_dim,
                               causal):
    if jt.get_compute_capability(0) < 80:
      raise unittest.SkipTest(
          "Fused attention only works on GPUs with capability >= sm80")
    k1, k2, k3 = random.split(random.PRNGKey(0), 3)
    q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim),
                      dtype=jnp.float16)
    k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim),
                      dtype=jnp.float16)
    v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim),
                      dtype=jnp.float16)

    def f(q, k, v):
      return attention.mha(q, k, v, causal=causal).sum()

    def f_ref(q, k, v):
      return attention.mha_reference(q, k, v, causal=causal).sum()

    dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
    dq_ref, dk_ref, dv_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)
    np.testing.assert_allclose(dq, dq_ref, atol=0.1)
    np.testing.assert_allclose(dk, dk_ref, atol=0.08)
    np.testing.assert_allclose(dv, dv_ref, atol=0.05)


class FusedLayerNormTest(parameterized.TestCase):

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_fused_layernorm_fwd(self, batch_size, seq_len, embed_dim):
    if jt.get_compute_capability(0) < 70:
      raise unittest.SkipTest(
          "Fused layernorm only works on GPUs with capability >= sm70")
    k1, k2, k3 = random.split(random.PRNGKey(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    o = layer_norm.layer_norm(x, w, b)
    o_ref = layer_norm.layer_norm_reference(x, w, b)
    np.testing.assert_allclose(o, o_ref, atol=1e-5)

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_fused_layernorm_bwd(self, batch_size, seq_len, embed_dim):
    if jt.get_compute_capability(0) < 70:
      raise unittest.SkipTest(
          "Fused layernorm only works on GPUs with capability >= sm70")
    k1, k2, k3 = random.split(random.PRNGKey(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    def f(x, w, b):
      return layer_norm.layer_norm(x, w, b).sum()

    def f_ref(x, w, b):
      return layer_norm.layer_norm_reference(x, w, b).sum()

    dx, dw, db = jax.grad(f, argnums=(0, 1, 2))(x, w, b)
    dx_ref, dw_ref, db_ref = jax.grad(f_ref, argnums=(0, 1, 2))(x, w, b)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(db, db_ref, rtol=1e-2, atol=1e-2)


class RmsNormTest(parameterized.TestCase):

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_rms_fwd(self, batch_size, seq_len, embed_dim):
    if jt.get_compute_capability(0) < 70:
      raise unittest.SkipTest(
          "Rms norm only works on GPUs with capability >= sm70")
    k1, k2, k3 = random.split(random.PRNGKey(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    o = rms_norm.rms_norm(x, w, b)
    o_ref = rms_norm.rms_norm_reference(x, w, b)
    np.testing.assert_allclose(o, o_ref, atol=1e-5)

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_rms_norm_bwd(self, batch_size, seq_len, embed_dim):
    if jt.get_compute_capability(0) < 70:
      raise unittest.SkipTest(
          "Rms norm only works on GPUs with capability >= sm70")
    k1, k2, k3 = random.split(random.PRNGKey(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    def f(x, w, b):
      return rms_norm.rms_norm(x, w, b).sum()

    def f_ref(x, w, b):
      return rms_norm.rms_norm_reference(x, w, b).sum()

    dx, dw, db = jax.grad(f, argnums=(0, 1, 2))(x, w, b)
    dx_ref, dw_ref, db_ref = jax.grad(f_ref, argnums=(0, 1, 2))(x, w, b)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(db, db_ref, rtol=1e-2, atol=1e-2)


class SoftmaxTest(parameterized.TestCase):

  @parameterized.parameters(
      (shape, dtype)
      for shape in [(1024, 125), (4, 1024, 125)]
      for dtype in (jnp.bfloat16, jnp.float16, jnp.float32)
  )
  def test_softmax(self, shape, dtype):
    # TODO(bchetioui): add Triton bug reference when filed
    if dtype == jnp.bfloat16:
      raise absltest.SkipTest("Disabled due to Triton lowering bug")

    x = jax.random.normal(random.PRNGKey(0), shape, dtype=dtype)

    atol, rtol = {
        jnp.bfloat16: (1e-2, 1e-4),
        jnp.float16: (1e-2, 1e-4),
        jnp.float32: (1e-7, 1e-6),
    }[dtype]

    np.testing.assert_allclose(
        softmax.softmax(x, axis=-1),
        jax.nn.softmax(x, axis=-1),
        atol=atol,
        rtol=rtol,
    )


if __name__ == "__main__":
  absltest.main()
