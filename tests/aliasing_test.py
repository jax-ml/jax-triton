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
Additional tests for input_output_aliases= parameter of triton_call().
"""


import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import config, random
import jax.numpy as jnp
import jax_triton as jt
import jax_triton.triton_lib as jttl
import numpy as np
import triton
import triton.language as tl

config.parse_flags_with_absl()


@triton.jit
def inc_inplace_kernel(x_in_out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_in_out_ptr + offsets, mask=mask)
  output = x + 1
  tl.store(x_in_out_ptr + offsets, output, mask=mask)


class TritonCallAliasingTest(parameterized.TestCase):
  @parameterized.parameters(False, True)
  def test_simple(self, with_donation):
    size = 8
    x = random.normal(random.key(0), [size])
    expected = x + 1

    launcher = lambda x: jt.triton_call(
      x,
      size,
      kernel=inc_inplace_kernel,
      out_shape=x,
      grid=(8,),
      BLOCK_SIZE=1,
      input_output_aliases={0: 0},
    )

    if with_donation:
      launcher = jax.jit(launcher, donate_argnums=(0,))
      x_ptr = x.unsafe_buffer_pointer()

    out = launcher(x)

    if with_donation:
      np.testing.assert_(x.is_deleted())
      np.testing.assert_equal(x_ptr, out.unsafe_buffer_pointer())

    np.testing.assert_allclose(out, expected)

  @parameterized.product(with_donation=[False, True], first_is_inout=[False, True])
  def test_2outputs(self, with_donation, first_is_inout):
    # this tests aliasing correctness in case of 4 buffer parameters two of which
    # (0 and 2, or 1 and 3) are in-out params. When first_is_inout=True, buffers 0 and 2
    # are incremented in place using values from buffers 1 and 3, and otherwise buffers
    # 1 and 3 are incremented in place with values from buffers 0 and 2.

    @triton.jit
    def weird_kernel(
      ptr0,
      ptr1,
      ptr2,
      ptr3,
      n_elements,
      BLOCK_SIZE: tl.constexpr,
      FIRST_INOUT: tl.constexpr,
    ):
      """
      For FIRST_INOUT=True  computes *ptr0[] += *ptr1[], *ptr2[] += *ptr3[],
      for FIRST_INOUT=False computes *ptr1[] += *ptr0[], *ptr3[] += *ptr2[]
      """
      pid = tl.program_id(axis=0)
      block_start = pid * BLOCK_SIZE
      offsets = block_start + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      if FIRST_INOUT:
        io0_ptr = ptr0
        io1_ptr = ptr2
        i0_ptr = ptr1
        i1_ptr = ptr3
      else:
        io0_ptr = ptr1
        io1_ptr = ptr3
        i0_ptr = ptr0
        i1_ptr = ptr2

      x0 = tl.load(io0_ptr + offsets, mask=mask)
      y0 = tl.load(i0_ptr + offsets, mask=mask)
      x1 = tl.load(io1_ptr + offsets, mask=mask)
      y1 = tl.load(i1_ptr + offsets, mask=mask)
      output0 = x0 + y0
      output1 = x1 + y1
      tl.store(io0_ptr + offsets, output0, mask=mask)
      tl.store(io1_ptr + offsets, output1, mask=mask)

    size = 8
    keys = random.split(random.key(0), 4)
    args = [random.normal(key, [size]) for key in keys]
    expect0 = args[0] + args[1]
    expect1 = args[2] + args[3]

    inout_aliases = {0: 0, 2: 1} if first_is_inout else {1: 0, 3: 1}
    inout_indices = tuple(inout_aliases.keys())
    in_indices = (1, 3) if first_is_inout else (0, 2)

    launcher = lambda *a: jt.triton_call(
      *a,
      size,
      kernel=weird_kernel,
      out_shape=tuple(a[io] for io in inout_indices),
      input_output_aliases=inout_aliases,
      grid=(8,),
      BLOCK_SIZE=1,
      FIRST_INOUT=first_is_inout,
    )
    if with_donation:
      launcher = jax.jit(launcher, donate_argnums=inout_indices)
      io_ptrs = tuple(args[io].unsafe_buffer_pointer() for io in inout_indices)

    outs = launcher(*args)

    if with_donation:
      for io in inout_indices:
        np.testing.assert_(args[io].is_deleted())
      for i in in_indices:
        np.testing.assert_(not args[i].is_deleted())
      np.testing.assert_array_equal(
        io_ptrs, tuple(o.unsafe_buffer_pointer() for o in outs)
      )

    np.testing.assert_allclose(outs[0], expect0)
    np.testing.assert_allclose(outs[1], expect1)

  @parameterized.parameters(0, "x_in_out_ptr")
  def test_raw_values(self, input_output_aliases):
    """test raw values in input_output_aliases (without a container wrapping)."""
    size = 8
    x = random.normal(random.key(0), [size])
    out = jt.triton_call(
      x,
      size,
      kernel=inc_inplace_kernel,
      grid=(8,),
      BLOCK_SIZE=1,
      input_output_aliases=input_output_aliases,
    )
    np.testing.assert_array_equal(out, x + 1)

  @parameterized.parameters([("Ptrs", 1)], [(0, 1)], (("Ptrs", 1),), ((0, 1),))
  def test_tuple_coordinate_into_compound(self, alias_spec):
    """Tuple coordinate drilling into a specific element of a compound param.
    Tests both ("name", idx) and (int, idx) forms."""

    @jt.kernel
    @triton.jit
    def _k(Ptrs, val_ptr, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      v = tl.load(val_ptr + offs)
      x = tl.load(Ptrs[1] + offs)
      tl.store(Ptrs[1] + offs, x + v)

    a = jnp.array([10.0], dtype=jnp.float32)
    b = jnp.array([20.0], dtype=jnp.float32)
    val = jnp.array([5.0], dtype=jnp.float32)
    out = _k[(1,)]((a, b), val, 1, input_output_aliases=[alias_spec])
    np.testing.assert_array_equal(out, jnp.array([25.0]))

  def test_nested_list_structured_output(self):
    """input_output_aliases=[0, [1, 2]] returns (flat, (nested, nested)) and
    related tests"""

    @jt.kernel
    @triton.jit
    def _k(ptr0, ptr1, ptr2, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      tl.store(ptr0 + offs, tl.load(ptr0 + offs) + 1)
      tl.store(ptr1 + offs, tl.load(ptr1 + offs) + 2)
      tl.store(ptr2 + offs, tl.load(ptr2 + offs) + 3)

    a = jnp.array([10.0], dtype=jnp.float32)
    b = jnp.array([20.0], dtype=jnp.float32)
    c = jnp.array([30.0], dtype=jnp.float32)

    result = _k[1](a, b, c, BLOCK=1, input_output_aliases=[0, [1, 2]])
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)
    np.testing.assert_array_equal(result[0], jnp.array([11.0]))
    self.assertIsInstance(result[1], tuple)
    self.assertEqual(len(result[1]), 2)
    np.testing.assert_array_equal(result[1][0], jnp.array([22.0]))
    np.testing.assert_array_equal(result[1][1], jnp.array([33.0]))

    result = _k[1](a, b, c, BLOCK=1, input_output_aliases=[2, [0, 1]])
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)
    np.testing.assert_array_equal(result[0], jnp.array([33.0]))
    self.assertIsInstance(result[1], tuple)
    self.assertEqual(len(result[1]), 2)
    np.testing.assert_array_equal(result[1][0], jnp.array([11.0]))
    np.testing.assert_array_equal(result[1][1], jnp.array([22.0]))
    assert jttl.JTJITFunction(_k).compiled_kernels_cache_size == 1

    result = _k[1](a, b, c, BLOCK=1, input_output_aliases=[[1, 0], 2])
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)
    np.testing.assert_array_equal(result[1], jnp.array([33.0]))
    self.assertIsInstance(result[0], tuple)
    self.assertEqual(len(result[0]), 2)
    np.testing.assert_array_equal(result[0][0], jnp.array([22.0]))
    np.testing.assert_array_equal(result[0][1], jnp.array([11.0]))
    assert jttl.JTJITFunction(_k).compiled_kernels_cache_size == 1

  @parameterized.parameters(
    "inout_ptr",
    ["inout_ptr"],
    ("inout_ptr",),
    (("inout_ptr",),),
    [("inout_ptr",)],
    1,
    (1,),
    [1],
    [(1,)],
    ((1,),),
  )
  def test_mixed_pure_and_aliased_outputs(self, input_output_aliases):
    """Pure output + aliased output in a single call.
    Exercises the return assembly: tuple(pure_outs.values()) + aliased_outs."""

    @jt.kernel(out_names="out_ptr")
    @triton.jit
    def _k(in_ptr, inout_ptr, out_ptr, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      x = tl.load(in_ptr + offs)
      y = tl.load(inout_ptr + offs)
      tl.store(out_ptr + offs, x * 2)
      tl.store(inout_ptr + offs, y + x)

    x = jnp.array([5.0], dtype=jnp.float32)
    y = jnp.array([10.0], dtype=jnp.float32)
    pure_out, aliased_out = _k[(1,)](
      x, y, BLOCK=1, out_shape=x, input_output_aliases=input_output_aliases
    )
    np.testing.assert_array_equal(pure_out, x * 2)
    np.testing.assert_array_equal(aliased_out, y + x)

  @parameterized.parameters(
    "inout_ptr",
    ["inout_ptr"],
    ("inout_ptr",),
    (("inout_ptr",),),
    [("inout_ptr",)],
    1,
    (1,),
    [1],
    [(1,)],
    ((1,),),
  )
  def test_mixed_multiple_pure_and_aliased_outputs(self, input_output_aliases):
    """2 pure outputs + 1 aliased output, verifying ordering."""

    @jt.kernel(out_names=("out1_ptr", "out2_ptr"))
    @triton.jit
    def _k(in_ptr, inout_ptr, out1_ptr, out2_ptr, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      x = tl.load(in_ptr + offs)
      y = tl.load(inout_ptr + offs)
      tl.store(out1_ptr + offs, x * 2)
      tl.store(out2_ptr + offs, x * 3)
      tl.store(inout_ptr + offs, y + x)

    x = jnp.array([5.0], dtype=jnp.float32)
    y = jnp.array([10.0], dtype=jnp.float32)
    out1, out2, aliased_out = _k[(1,)](
      x, y, BLOCK=1, out_shape=(x, x), input_output_aliases=input_output_aliases
    )
    np.testing.assert_array_equal(out1, x * 2)
    np.testing.assert_array_equal(out2, x * 3)
    np.testing.assert_array_equal(aliased_out, y + x)

  @parameterized.product(
    out_shape_key=["out_ptr", 2, ("out_ptr",), (2,)],
    input_output_aliases=[
      "inout_ptr",
      ["inout_ptr"],
      ("inout_ptr",),
      (("inout_ptr",),),
      [("inout_ptr",)],
      1,
      (1,),
      [1],
      [(1,)],
      ((1,),),
    ],
  )
  def test_with_dict_out_shape(self, out_shape_key, input_output_aliases):
    """Dict out_shape combined with sequence-form input_output_aliases.
    The dict provides pure output shapes; aliased shapes are inferred from inputs."""

    @jt.kernel
    @triton.jit
    def _k(in_ptr, inout_ptr, out_ptr, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      x = tl.load(in_ptr + offs)
      y = tl.load(inout_ptr + offs)
      tl.store(out_ptr + offs, x * 2)
      tl.store(inout_ptr + offs, y + x)

    x = jnp.array([5.0], dtype=jnp.float32)
    y = jnp.array([10.0], dtype=jnp.float32)
    pure_out, aliased_out = _k[(1,)](
      x,
      y,
      BLOCK=1,
      out_shape={out_shape_key: x},
      input_output_aliases=input_output_aliases,
    )
    np.testing.assert_array_equal(pure_out, x * 2)
    np.testing.assert_array_equal(aliased_out, y + x)

  @parameterized.parameters(
    "inout_ptr",
    ("inout_ptr",),
    (("inout_ptr",),),
    1,
    (1,),
    ((1,),),
  )
  def test_dict_form_with_pure_and_aliased_out_shape(self, aliasing_key):
    """Deprecated dict-form aliases where out_shape carries both pure and
    aliased shapes, split by out_names count."""

    @jt.kernel(out_names="out_ptr")
    @triton.jit
    def _k(in_ptr, inout_ptr, out_ptr, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      x = tl.load(in_ptr + offs)
      y = tl.load(inout_ptr + offs)
      tl.store(out_ptr + offs, x * 2)
      tl.store(inout_ptr + offs, y + x)

    x = jnp.array([5.0], dtype=jnp.float32)
    y = jnp.array([10.0], dtype=jnp.float32)
    pure_out, aliased_out = _k[1](
      x,
      y,
      out_shape=(x, y),  # the first is pure output, the second is aliased
      BLOCK=1,
      input_output_aliases={aliasing_key: 1},
    )
    np.testing.assert_array_equal(pure_out, x * 2)
    np.testing.assert_array_equal(aliased_out, y + x)

  def test_compound_with_mixed_dtypes(self):
    """String aliasing a compound param whose arrays have different dtypes.
    Verifies functools.reduce(_unpack_arg, ...) creates correct per-dtype shapes."""

    @jt.kernel
    @triton.jit
    def _k(Ptrs, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      x = tl.load(Ptrs[0] + offs)
      y = tl.load(Ptrs[1] + offs)
      tl.store(Ptrs[0] + offs, x + 1)
      tl.store(Ptrs[1] + offs, y + 10)

    f = jnp.array([5.0], dtype=jnp.float32)
    i = jnp.array([100], dtype=jnp.int32)
    out = _k[(1,)]((f, i), 1, input_output_aliases="Ptrs")
    self.assertIsInstance(out, tuple)
    self.assertEqual(len(out), 2)
    np.testing.assert_array_equal(out[0], jnp.array([6.0], dtype=jnp.float32))
    self.assertEqual(out[0].dtype, jnp.float32)
    np.testing.assert_array_equal(out[1], jnp.array([110], dtype=jnp.int32))
    self.assertEqual(out[1].dtype, jnp.int32)


if __name__ == "__main__":
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  absltest.main()
