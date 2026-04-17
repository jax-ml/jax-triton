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
Additional tests for out_shape= parameter of triton_call().
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
from jax import config
import jax.numpy as jnp
import jax_triton as jt
import jax_triton.triton_lib as jttl
import numpy as np
import triton
import triton.language as tl

config.parse_flags_with_absl()


class TritonCallOutShapeTest(parameterized.TestCase):
  def test_single_output(self):
    """All forms of out_shape for a single output parameter."""

    @jt.kernel
    @triton.jit
    def copy_k(in_ptr, n_elements, out_ptr, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(axis=0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offs < n_elements
      tl.store(out_ptr + offs, tl.load(in_ptr + offs, mask=mask), mask=mask)

    def assert_expected(out):
      np.testing.assert_array_equal(out, x)
      assert jttl.JTJITFunction(copy_k).compiled_kernels_cache_size == 1

    x = jnp.arange(16, dtype=jnp.float32)

    for out_shape in [{"out_ptr": x}, {2: x}, x, [x], (x,)]:
      for out_names in [["out_ptr"], ("out_ptr",), "out_ptr", 2, [2], (2,), None]:
        if (
          isinstance(out_shape, dict)
          and out_names is not None
          and frozenset(out_shape.keys())
          != frozenset(
            out_names if isinstance(out_names, (tuple, list)) else (out_names,)
          )
        ):
          out_names = None
        out = copy_k[(2,)](
          x, x.size, BLOCK_SIZE=8, out_shape=out_shape, out_names=out_names
        )
        assert_expected(out)

  def test_multiple_outputs(self):
    """All forms of out_shape for multiple output parameters."""

    @jt.kernel
    @triton.jit
    def twin_k(in_ptr, n, out1_ptr, out2_ptr, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(axis=0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offs < n
      x = tl.load(in_ptr + offs, mask=mask)
      tl.store(out1_ptr + offs, x, mask=mask)
      tl.store(out2_ptr + offs, x + 1, mask=mask)

    def assert_expected(o1, o2):
      np.testing.assert_array_equal(o1, x)
      np.testing.assert_array_equal(o2, x + 1)
      assert jttl.JTJITFunction(twin_k).compiled_kernels_cache_size == 1

    x = jnp.arange(8, dtype=jnp.float32)

    for out_shape in [
      {"out1_ptr": x, "out2_ptr": x},
      {"out2_ptr": x, "out1_ptr": x},
      {2: x, "out2_ptr": x},
      {3: x, "out1_ptr": x},
      {"out1_ptr": x, 3: x},
      {"out2_ptr": x, 2: x},
      {2: x, 3: x},
      {3: x, 2: x},
      (x, x),
      [x, x],
    ]:
      for out_names in [
        None,
        ("out1_ptr", "out2_ptr"),
        ("out2_ptr", "out1_ptr"),
        [2, 3],
        [3, 2],
        [2, "out2_ptr"],
        [3, "out1_ptr"],
        (2, 3),
        (3, 2),
        ("out1_ptr", 3),
        (2, "out2_ptr"),
      ]:
        if (
          isinstance(out_shape, dict)
          and out_names is not None
          and frozenset(out_shape.keys())
          != frozenset(
            out_names if isinstance(out_names, (tuple, list)) else (out_names,)
          )
        ):
          out_names = None
        o1, o2 = twin_k[(1,)](
          x, x.size, BLOCK_SIZE=8, out_shape=out_shape, out_names=out_names
        )
        assert_expected(o1, o2)

  def test_different_shapes(self):
    """Multiple outputs with different shapes via sequence form."""

    @jt.kernel
    @triton.jit
    def split_k(in_ptr, out1_ptr, out2_ptr, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(axis=0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask8 = offs < 8
      mask7 = offs < 7
      tl.store(out1_ptr + offs, tl.load(in_ptr + offs, mask=mask8), mask=mask8)
      tl.store(out2_ptr + offs, tl.load(in_ptr + 8 + offs, mask=mask7), mask=mask7)

    x = jnp.arange(15, dtype=jnp.float32)
    o1, o2 = split_k[(1,)](x, BLOCK_SIZE=8, out_shape=[x[:8], x[8:]])
    np.testing.assert_array_equal(o1, x[:8])
    np.testing.assert_array_equal(o2, x[8:])

  def test_different_dtypes(self):
    """Multiple outputs with different dtypes."""

    @jt.kernel
    @triton.jit
    def cast_k(in_ptr, out_f32_ptr, out_i32_ptr):
      x = tl.load(in_ptr)
      tl.store(out_f32_ptr, x)
      tl.store(out_i32_ptr, x.to(tl.int32))

    x = jnp.array([42.25], dtype=jnp.float32)
    o_f, o_i = cast_k[(1,)](x, out_shape=[x, x.astype(jnp.int32)])
    np.testing.assert_array_equal(o_f, x)
    self.assertEqual(o_f.dtype, jnp.float32)
    np.testing.assert_array_equal(o_i, jnp.array([42], dtype=jnp.int32))
    self.assertEqual(o_i.dtype, jnp.int32)

  def test_dict_ordering(self):
    """Dict out_shape + explicit out_names: ordering doesn't matter and is always
    governed by the kernel signature."""

    @jt.kernel
    @triton.jit
    def twin_k(in_ptr, out_a_ptr, out_b_ptr):
      x = tl.load(in_ptr)
      tl.store(out_a_ptr, x.to(tl.int32))
      tl.store(out_b_ptr, x + 10)

    x = jnp.array([5.25], dtype=jnp.float32)

    def assert_expected(o1, o2):
      np.testing.assert_array_equal(o1, x.astype(jnp.int32))
      self.assertEqual(o1.dtype, jnp.int32)
      np.testing.assert_array_equal(o2, x + 10)
      self.assertEqual(o2.dtype, jnp.float32)
      assert jttl.JTJITFunction(twin_k).compiled_kernels_cache_size == 1

    for out_shape in [
      {"out_b_ptr": x, "out_a_ptr": x.astype(jnp.int32)},
      {"out_a_ptr": x.astype(jnp.int32), "out_b_ptr": x},
    ]:
      for out_names in [
        ("out_b_ptr", "out_a_ptr"),
        ("out_a_ptr", "out_b_ptr"),
        None,
      ]:
        o1, o2 = twin_k[(1,)](x, out_shape=out_shape, out_names=out_names)
        assert_expected(o1, o2)

  def test_compound_first(self):
    """out_shape=((a,(b,c)),d) — first output param is a compound tuple of arrays."""

    @jt.kernel
    @triton.jit
    def compound_k(in_ptr, Ptrs, single_out):
      x = tl.load(in_ptr)
      tl.static_assert(len(Ptrs) == 2, "Ptrs must be a tuple of 2 elements")
      tl.store(Ptrs[0], x)
      tl.static_assert(len(Ptrs[1]) == 2, "Ptrs[1] must be a tuple of 2 arrays")
      tl.store(Ptrs[1][0], x.to(tl.int32) + 1)
      tl.store(Ptrs[1][1], x.to(tl.int16) + 2)
      tl.store(single_out, x * 2)

    x = jnp.array([5.25])
    z = jnp.zeros((1,), dtype=jnp.float32)

    compound_out, scalar_out = compound_k[(1,)](
      x, out_shape=((z, (z.astype(jnp.int32), z.astype(jnp.int16))), z)
    )
    self.assertIsInstance(compound_out, tuple)
    self.assertEqual(len(compound_out), 2)
    np.testing.assert_array_equal(compound_out[0], x)
    self.assertEqual(compound_out[0].dtype, jnp.float32)

    self.assertIsInstance(compound_out[1], tuple)
    self.assertEqual(len(compound_out[1]), 2)
    np.testing.assert_array_equal(compound_out[1][0], (x + 1).astype(jnp.int32))
    self.assertEqual(compound_out[1][0].dtype, jnp.int32)
    np.testing.assert_array_equal(compound_out[1][1], (x + 2).astype(jnp.int16))
    self.assertEqual(compound_out[1][1].dtype, jnp.int16)

    np.testing.assert_array_equal(scalar_out, x * 2)
    self.assertEqual(scalar_out.dtype, jnp.float32)

  def test_compound_second(self):
    """out_shape=(a, (b,(c,d),e)) — second output param is a compound tuple of arrays."""

    @jt.kernel
    @triton.jit
    def compound_k(in_ptr, single_out, Ptrs):
      x = tl.load(in_ptr)
      tl.store(single_out, x * 2)
      tl.static_assert(len(Ptrs) == 3, "Ptrs must be a tuple of 3 elements")
      tl.static_assert(len(Ptrs[1]) == 2, "Ptrs[1] must be a tuple of 2 arrays")
      tl.store(Ptrs[0], x)
      tl.store(Ptrs[1][0], x.to(tl.int32) + 1)
      tl.store(Ptrs[1][1], x.to(tl.int16) + 2)
      tl.store(Ptrs[2], x.to(tl.int8) + 3)

    x = jnp.array([5.25])
    z = jnp.zeros((1,), dtype=jnp.float32)
    scalar_out, compound_out = compound_k[(1,)](
      x,
      out_shape=(
        z,
        (z, (z.astype(jnp.int32), z.astype(jnp.int16)), z.astype(jnp.int8)),
      ),
    )
    np.testing.assert_array_equal(scalar_out, x * 2)
    self.assertEqual(scalar_out.dtype, jnp.float32)

    self.assertIsInstance(compound_out, tuple)
    self.assertEqual(len(compound_out), 3)
    self.assertIsInstance(compound_out[1], tuple)
    self.assertEqual(len(compound_out[1]), 2)

    np.testing.assert_array_equal(compound_out[0], x)
    self.assertEqual(compound_out[0].dtype, jnp.float32)

    np.testing.assert_array_equal(compound_out[1][0], (x + 1).astype(jnp.int32))
    self.assertEqual(compound_out[1][0].dtype, jnp.int32)
    np.testing.assert_array_equal(compound_out[1][1], (x + 2).astype(jnp.int16))
    self.assertEqual(compound_out[1][1].dtype, jnp.int16)

    np.testing.assert_array_equal(compound_out[2], (x + 3).astype(jnp.int8))
    self.assertEqual(compound_out[2].dtype, jnp.int8)

  def test_dict_compound(self):
    """Dict form: out_shape={"Ptrs": ((a,b),c), "single_out": d} — compound dict value."""

    @jt.kernel
    @triton.jit
    def compound_k(in_ptr, Ptrs, single_out):
      x = tl.load(in_ptr)
      tl.static_assert(len(Ptrs) == 2, "Ptrs must be a tuple of 2 elements")
      tl.static_assert(len(Ptrs[0]) == 2, "Ptrs[0] must be a tuple of 2 arrays")
      tl.store(Ptrs[0][0], x)
      tl.store(Ptrs[0][1], x.to(tl.int32) + 1)
      tl.store(Ptrs[1], x.to(tl.int16) + 2)
      tl.store(single_out, x * 2)

    x = jnp.array([5.25])
    z = jnp.zeros((1,), dtype=jnp.float32)

    def assert_expected(compound_out, scalar_out):
      self.assertIsInstance(compound_out, tuple)
      self.assertEqual(len(compound_out), 2)
      self.assertIsInstance(compound_out[0], tuple)
      self.assertEqual(len(compound_out[0]), 2)

      np.testing.assert_array_equal(compound_out[0][0], x)
      self.assertEqual(compound_out[0][0].dtype, jnp.float32)
      np.testing.assert_array_equal(compound_out[0][1], (x + 1).astype(jnp.int32))
      self.assertEqual(compound_out[0][1].dtype, jnp.int32)

      np.testing.assert_array_equal(compound_out[1], (x + 2).astype(jnp.int16))
      self.assertEqual(compound_out[1].dtype, jnp.int16)

      np.testing.assert_array_equal(scalar_out, x * 2)
      self.assertEqual(scalar_out.dtype, jnp.float32)

      assert jttl.JTJITFunction(compound_k).compiled_kernels_cache_size == 1

    compound_out, scalar_out = compound_k[(1,)](
      x,
      out_shape={
        "Ptrs": ((z, z.astype(jnp.int32)), z.astype(jnp.int16)),
        "single_out": z,
      },
    )
    assert_expected(compound_out, scalar_out)

    # Reversed dict key ordering — result must be identical (kernel signature order wins)
    compound_out2, scalar_out2 = compound_k[(1,)](
      x,
      out_shape={
        "single_out": z,
        "Ptrs": ((z, z.astype(jnp.int32)), z.astype(jnp.int16)),
      },
    )
    assert_expected(compound_out2, scalar_out2)

  def test_integer_interleaved(self):
    """Integer out_names for output params interleaved with a kwarg-only scalar input."""

    @jt.kernel
    @triton.jit
    def interleaved_k(in_ptr, out1_ptr, scale, out2_ptr):
      x = tl.load(in_ptr)
      tl.store(out1_ptr, x)
      tl.store(out2_ptr, x * scale)

    x = jnp.array([5.25])
    z = jnp.zeros((1,), dtype=jnp.float32)

    o1, o2 = interleaved_k[(1,)](x, out_shape=[z, z], out_names=(1, 3), scale=3.0)
    np.testing.assert_array_equal(o1, x)
    np.testing.assert_array_equal(o2, x * 3)


if __name__ == "__main__":
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  absltest.main()
