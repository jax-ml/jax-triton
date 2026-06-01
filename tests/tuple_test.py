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

"""
Additional tests for tuple passing and returning.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import config, random
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl
from typing import NamedTuple

config.parse_flags_with_absl()


def jax_strides(x):
  assert isinstance(x, jax.Array) and x.ndim == 2
  return (x.shape[1], 1)


# many tests here are inspired by the original Triton's test suite


class TupleTest(parameterized.TestCase):
  @parameterized.parameters(1, 2, 3, 4)
  # the original test also tested size=0, however, this leads to an invocation of the
  # kernel with no outputs, which triggers JAX's DCE. DCE could be disabled for a
  # kernel call if needed, but the utility of that isn't exactly clear, so until
  # clarified, size=0 is excluded.
  def test_index(self, size):
    @triton.jit
    def _tuple_increment(values):
      return tl.tuple([v + 1 for v in values])

    @triton.jit
    def _tuple_index_func(Ptrs, values):
      for i in tl.static_range(len(values)):
        tl.store(Ptrs[i], values[i])

    @jt.kernel(out_names="Ptrs")
    @triton.jit
    def _tuple_index(_0, _1: tl.constexpr, values, _2, _3: tl.constexpr, _4, Ptrs):
      values = _tuple_increment(values)
      _tuple_index_func(Ptrs, values)

    vals = tuple(i + 1 for i in range(size))
    rets = tuple(jnp.zeros((1,), dtype=jnp.float32) for _ in vals)
    rets = _tuple_index[(1,)](0, 0, vals, 0, 0, 0, out_shape=[rets])
    assert vals == tuple(x - 1 for x in rets)

  def test_assign(self):
    @jt.kernel
    @triton.jit
    def _tuple_assign(XPtrs, YPtrs, values):
      # assign from tuple
      X0, X1 = XPtrs
      x0, x1, _ = values
      tl.store(X0, x0)
      tl.store(X1, x1)
      # assign to tuple
      Y0, Y1, Y2 = YPtrs
      Y = Y0, Y1, Y2
      y = x0, 10, x1
      tl.store(Y[0], y[0])
      tl.store(Y[1], y[1])
      tl.store(Y[2], y[2])

    vals = (2.0, 3.0, None)
    x = tuple(jnp.zeros((1,), dtype=jnp.float32) for _ in range(2))
    y = tuple(jnp.zeros((1,), dtype=jnp.float32) for _ in range(3))
    x, y = _tuple_assign[(1,)](x, y, vals, input_output_aliases=["XPtrs", "YPtrs"])
    assert x[0] == vals[0]
    assert x[1] == vals[1]
    assert y[0] == vals[0]
    assert y[1] == 10
    assert y[2] == vals[1]

  def test_assign_return(self):
    @triton.jit
    def _tuple_ret(a, b):
      return a + b, a - b, a * b

    @jt.kernel
    @triton.jit
    def with_fn(X, Y, A, B, C):
      x = tl.load(X)
      y = tl.load(Y)
      a, b, c = _tuple_ret(x, y)
      tl.store(A, a)
      tl.store(B, b)
      tl.store(C, c)

    @jt.kernel
    @triton.jit
    def without_fn(X, Y, A, B, C):
      x = tl.load(X)
      y = tl.load(Y)
      a, b, c = x + y, x - y, x * y
      tl.store(A, a)
      tl.store(B, b)
      tl.store(C, c)

    x = jnp.array([1.3], dtype=jnp.float32)
    y = jnp.array([1.9], dtype=jnp.float32)
    for kernel in [with_fn, without_fn]:
      a_tri, b_tri, c_tri = kernel[(1,)](x, y, num_warps=1, out_shape=(x, x, x))
      a_ref, b_ref, c_ref = x + y, x - y, x * y
      assert a_tri == a_ref
      assert b_tri == b_ref
      assert c_tri == c_ref

  def test_serialize(self):
    @triton.jit
    def _tuple_fn0(Ptr, cst2: tl.constexpr, tuple1):
      tl.static_assert(tuple1[1] is None)
      tl.store(Ptr + 5, cst2)
      tl.store(Ptr + 6, tuple1[0])
      tl.store(Ptr + 7, tl.load(tuple1[2][0]))
      tl.store(Ptr + 8, tuple1[2][1][0])
      tl.store(Ptr + 9, tl.load(tuple1[2][1][2]))

    @jt.kernel(out_names="Ptr")
    @triton.jit
    def _tuple_serialize(N1, tuple1, cst1: tl.constexpr, val1, tuple2, Ptr):
      tl.static_assert(N1 is None)
      tl.static_assert(tuple1[1][1] is None)
      tl.static_assert(tuple1[1][3] == 4)
      tl.store(Ptr + 0, tl.load(tuple1[0]))
      tl.store(Ptr + 1, tuple1[1][0])
      tl.store(Ptr + 2, tl.load(tuple1[1][2]))
      tl.store(Ptr + 3, cst1 + val1)
      tl.store(Ptr + 4, tl.load(tuple2[0]))
      _tuple_fn0(Ptr, 15, (-1, None, tuple1))

    x0 = jnp.array([8], dtype=jnp.int32)
    x1 = jnp.array([12], dtype=jnp.int32)
    y0 = jnp.array([10], dtype=jnp.int32)

    z = _tuple_serialize[(1,)](
      None,
      (x0, (1, None, x1, tl.constexpr(4))),
      20,
      1,
      (y0,),
      out_shape=jnp.empty((10,), dtype=jnp.int32),
    )
    ref = jnp.array([8, 1, 12, 21, 10, 15, -1, 8, 1, 12], dtype=jnp.int32)
    np.testing.assert_array_equal(z, ref)

  def test_namedtuple(self):
    class Function(NamedTuple):
      fn: tl.constexpr
      captured: tuple

    class Tensor(NamedTuple):
      ptr: any
      shape: tuple
      stride: tuple

    @triton.jit
    def _namedtuple_create_func0(shape, ptr, stride):
      return Tensor(shape=shape, ptr=ptr, stride=stride)

    @triton.jit
    def _namedtuple_create_func1(shape, ptr, stride):
      tensor = Tensor(shape=shape, ptr=ptr, stride=stride)
      return tensor

    @triton.jit
    def _namedtuple_mask_func(Tensor, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      offs_m = tl.arange(0, BLOCK_M)
      offs_n = tl.arange(0, BLOCK_N)
      mask = (offs_m[:, None] < Tensor.shape[0]) & (offs_n[None, :] < Tensor.shape[1])
      return mask

    @jt.kernel
    @triton.jit
    def _namedtuple_kernel(
      closure, _X, Y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
      offs_m = tl.arange(0, BLOCK_M)
      offs_n = tl.arange(0, BLOCK_N)
      X = _namedtuple_create_func0(_X.shape, _X.ptr, _X.stride)
      Y = _namedtuple_create_func1(Y.shape, Y.ptr, Y.stride)
      Xs = X.ptr + offs_m[:, None] * X.stride[0] + offs_n[None, :] * X.stride[1]
      Ys = Y.ptr + offs_m[:, None] * Y.stride[0] + offs_n[None, :] * Y.stride[1]
      x = tl.load(Xs, mask=_namedtuple_mask_func(X, BLOCK_M, BLOCK_N), other=0)
      y = closure.fn(x, *closure.captured)
      tl.store(Ys, y, mask=_namedtuple_mask_func(Y, BLOCK_M, BLOCK_N))

    x = random.normal(random.key(0), (32, 32), dtype=jnp.float32)
    y = jnp.empty((16, 16), dtype=jnp.float32)
    a = jnp.array([5.2], dtype=jnp.float32)

    @triton.jit
    def mul(x, a):
      return x * tl.load(a)

    function = Function(mul, (a,))
    tx = Tensor(x, x.shape, jax_strides(x))
    ty = Tensor(y, y.shape, jax_strides(y))
    y = _namedtuple_kernel[(1,)](function, tx, ty, 64, 64, input_output_aliases="Y")
    np.testing.assert_allclose(y, x[:16, :16] * a)

  def test_passing_nested_tuple_with_constexpr(self):
    @jt.kernel
    @triton.jit
    def _nested_tuple_kernel(x, out_ptr):
      # This creates a new scope, which will force a copy of liveins. It's
      # important for this to happen as it forces IR flattening/unflattening,
      # which relies on the types being correct for the roundtrip to succeed.
      for _ in range(1):
        tl.static_assert(x[1][0] == 2)
        tl.static_assert(len(x[2]) == 0)  # to tests JT specific empty tuple passing

    _nested_tuple_kernel[(1,)](  # out_shape is needed to prevent DCE
      ((1,), (tl.constexpr(2),), tuple()), out_shape=jnp.zeros(1)
    )

  def test_passing_tuple_to_make_tensor_descriptor(self):
    @jt.kernel
    @triton.jit
    def m_to_the_n(X_base, shape, strides, m_n, BLOCK_DIM: tl.constexpr):
      tl.static_assert(isinstance(strides[1].type, tl.constexpr_type))
      X = tl.make_tensor_descriptor(
        X_base,
        shape=shape,
        strides=strides,
        block_shape=[BLOCK_DIM, BLOCK_DIM],
      )
      # Make sure tl.make_tensor_descriptor didn't modify strides (i.e. didn't unwrap the constexpr)
      tl.static_assert(isinstance(strides[1].type, tl.constexpr_type))
      data = X.load([0, 0])
      # Include a for loop to ensure strides[1] is lifted into a constexpr
      # (otherwise cloning the local scope will fail).
      for i in tl.range(0, m_n[1]):
        data = m_n[0] * data
      X.store([0, 0], data)

    x = jnp.arange(0, 16).reshape(4, 4)
    expected_x = 8 * x.copy()
    x = m_to_the_n[(1,)](
      x, x.shape, jax_strides(x), (2, 3), x.shape[0], input_output_aliases="X_base"
    )
    np.testing.assert_array_equal(x, expected_x)


if __name__ == "__main__":
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  absltest.main()
