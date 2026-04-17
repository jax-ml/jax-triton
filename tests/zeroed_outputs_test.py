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
Additional tests for zeroed_outputs= parameter of triton_call().

Note that all these tests have an inherent weakness that could render them potentially
useless should the backend behavior change: zeroing of purely output arguments is
tricky, since when the backend initially allocates memory on a device, the memory comes
from a driver always already zeroed due to security concerns. The test are based on an
observed backend behavior that if the same kernel is launched 2 times while the outputs
of the first run are discarded, then on the next run kernel's purely output argument
will retain its old "dirty" values. If this behavior changes, the tests will become
flaky.

Previously the only test for zeroed_outputs= was based on zeroing an in-out argument.
The problem with that is that zeroing in-out arguments doesn't seem to have any
real-world use, since it just turns an in-out argument back into purely output argument,
all while likely still requiring the backend to copy the argument content from the host
to a device before clearing it. This prevents any information from passing into the
kernel via the aliased argument. Implementing that feature with a proper support for
tuples is cumbersome and does not worth the effort considering it has no real-world use
beyond testing.

Unfortunately, I don't have better ideas at this time how to test this in a more robust
fashion.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import config, random
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl

config.parse_flags_with_absl()


ERR_MSG = (
  "This test's assumption that subsequent calls use dirty buffers is "
  "violated (likely because of the backend behavior change). This makes the test "
  "either flaky or useless. Anyway, the test needs to be fixed and until that the "
  "best possible solution is to disable/skip this test."
)

# large array is likely to be reallocated to the same memory
ARRAY_SIZE = 100000


@jt.kernel
@triton.jit
def zeroing_kernel(
  x_ptr: tl.const,
  n_elements,
  out_ptr,
  BLOCK_SIZE: tl.constexpr,
  CLEANUP: tl.constexpr = False,  # this also implicitly test defaults
):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  # explicit cleanup is needed to isolate the kernel from previous tests run history
  if CLEANUP:
    tl.store(out_ptr + offsets, 0, mask=mask)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(out_ptr + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)


class TritonCallZeroedOutputsTest(parameterized.TestCase):
  @parameterized.product(
    use_function=[False, True], arg_form=[("out_ptr",), (2,), (("out_ptr",),), ((2,),)]
  )
  def test_arg_form_raw(self, use_function, arg_form):
    x = random.normal(random.key(0), shape=ARRAY_SIZE)
    BLOCK_SIZE = 8
    grid = triton.cdiv(x.size, BLOCK_SIZE)

    # first test the backend behavior, that sequential calls use dirty outputs.
    out = zeroing_kernel[grid](
      x, x.size, BLOCK_SIZE=BLOCK_SIZE, CLEANUP=True, out_shape=x
    )
    np.testing.assert_allclose(out, x)
    del out

    # and verify the core test assumption that second call uses a dirty buffer
    out = zeroing_kernel[grid](x, x.size, BLOCK_SIZE=BLOCK_SIZE, out_shape=x)
    np.testing.assert_allclose(out, 2 * x, err_msg=ERR_MSG)
    del out

    # finally test zeroed_outputs
    zeroed_arg = (lambda x: arg_form) if use_function else arg_form
    out = zeroing_kernel[grid](
      x, x.size, BLOCK_SIZE=BLOCK_SIZE, out_shape=x, zeroed_outputs=zeroed_arg
    )
    np.testing.assert_allclose(out, x)

  @parameterized.product(
    use_function=[False, True],
    zeroed_which=[
      # (zeroed_spec, which_zeroed) — which_zeroed tells us which tuple elements to expect zeroed
      (("out_ptrs", 0), {0}),  # string + sub-index, zero first only
      (("out_ptrs", 1), {1}),  # string + sub-index, zero second only
      ((2, 0), {0}),  # int + sub-index, zero first only
      ((2, 1), {1}),  # int + sub-index, zero second only
      ("out_ptrs", {0, 1}),  # string, zero both
      ((("out_ptrs", 0), ("out_ptrs", 1)), {0, 1}),  # tuples, zero both
      ((("out_ptrs", 1), ("out_ptrs", 0)), {0, 1}),  # tuples, zero both
      (2, {0, 1}),  # int, zero both
      (((2, 0), (2, 1)), {0, 1}),  # tuples, zero both
      (((2, 1), (2, 0)), {0, 1}),  # tuples, zero both
      (((2, 1), ("out_ptrs", 0)), {0, 1}),  # tuples, zero both
      ((("out_ptrs", 1), (2, 0)), {0, 1}),  # tuples, zero both
    ],
  )
  def test_tuple(self, use_function, zeroed_which):
    """zeroed_outputs uses a tuple path to address a single array inside a
    compound (tuple) output parameter."""
    zeroed_spec, which_zeroed = zeroed_which

    @jt.kernel(out_names="out_ptrs")
    @triton.jit
    def kernel_tuple_path(
      x_ptr: tl.const,
      n_elements,
      out_ptrs,
      BLOCK_SIZE: tl.constexpr,
      CLEANUP: tl.constexpr = False,
    ):
      pid = tl.program_id(axis=0)
      block_start = pid * BLOCK_SIZE
      offsets = block_start + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      for i in tl.static_range(len(out_ptrs)):
        if CLEANUP:
          tl.store(out_ptrs[i] + offsets, 0, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(out_ptrs[i] + offsets, mask=mask)
        tl.store(out_ptrs[i] + offsets, x + y, mask=mask)

    x = random.normal(random.key(20), shape=(ARRAY_SIZE,))
    BLOCK_SIZE = 8
    grid = triton.cdiv(x.size, BLOCK_SIZE)
    z = jax.ShapeDtypeStruct(x.shape, x.dtype)
    shape_pair = (z, z)

    args = [x, x.size]
    kwargs = dict(BLOCK_SIZE=BLOCK_SIZE, out_shape=[shape_pair])

    # prime
    o0, o1 = kernel_tuple_path[grid](*args, **kwargs, CLEANUP=True)
    np.testing.assert_allclose(o0, x)
    np.testing.assert_allclose(o1, x)
    del o0, o1
    # confirm dirty
    o0, o1 = kernel_tuple_path[grid](*args, **kwargs)
    np.testing.assert_allclose(o0, 2 * x, err_msg=ERR_MSG)
    np.testing.assert_allclose(o1, 2 * x, err_msg=ERR_MSG)
    del o0, o1

    # selective zeroing via tuple path
    zeroed_outputs = (
      (zeroed_spec,)
      if not isinstance(zeroed_spec, tuple) or not isinstance(zeroed_spec[0], tuple)
      else zeroed_spec
    )
    o0, o1 = kernel_tuple_path[grid](
      *args,
      **kwargs,
      zeroed_outputs=(lambda x: zeroed_outputs) if use_function else zeroed_outputs,
    )
    for i, out in enumerate((o0, o1)):
      if i in which_zeroed:
        np.testing.assert_allclose(out, x, err_msg=f"out_ptrs[{i}] should be zeroed")
      else:
        np.testing.assert_allclose(
          out, 3 * x, err_msg=f"out_ptrs[{i}] should NOT be zeroed"
        )

  @parameterized.parameters(True, False)
  def test_config_dependent_callable(self, zero_first):
    """Callable that inspects a constexpr metaparam to decide which output to zero.
    Also tests that modification of kwargs in callable is propagated."""

    @jt.kernel(out_names=("out0_ptr", "out1_ptr"))
    @triton.jit
    def kernel_configurable(
      x_ptr: tl.const,
      n_elements,
      out0_ptr,
      out1_ptr,
      BLOCK_SIZE: tl.constexpr,
      CLEANUP: tl.constexpr = False,
    ):
      pid = tl.program_id(axis=0)
      block_start = pid * BLOCK_SIZE
      offsets = block_start + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      if CLEANUP:
        tl.store(out0_ptr + offsets, 0, mask=mask)
        tl.store(out1_ptr + offsets, 0, mask=mask)
      x = tl.load(x_ptr + offsets, mask=mask)
      o0 = tl.load(out0_ptr + offsets, mask=mask)
      o1 = tl.load(out1_ptr + offsets, mask=mask)
      tl.store(out0_ptr + offsets, x + o0, mask=mask)
      tl.store(out1_ptr + offsets, x + o1, mask=mask)

    x = random.normal(random.key(50), shape=(ARRAY_SIZE,))
    BLOCK_SIZE = 8
    grid = triton.cdiv(x.size, BLOCK_SIZE)
    args = [x, x.size]
    kwargs = dict(BLOCK_SIZE=BLOCK_SIZE, out_shape=(x, x))

    # The callable inspects the WHICH constexpr from metaparams
    def zeroed_fn(meta):
      which_val = meta["WHICH"]
      del meta["WHICH"]  # also tests that modification of kwargs is propagated
      return ("out0_ptr",) if which_val == 0 else ("out1_ptr",)

    # prime both outputs to a dirty state
    o0, o1 = kernel_configurable[grid](*args, **kwargs, CLEANUP=True)
    np.testing.assert_allclose(o0, x)
    np.testing.assert_allclose(o1, x)
    del o0, o1
    o0, o1 = kernel_configurable[grid](*args, **kwargs)
    np.testing.assert_allclose(o0, 2 * x, err_msg=ERR_MSG)
    np.testing.assert_allclose(o1, 2 * x, err_msg=ERR_MSG)
    del o0, o1

    # WHICH=0 - callable zeros out0_ptr; WHICH=1 - callable zeros out1_ptr
    which_val = 0 if zero_first else 1
    o0, o1 = kernel_configurable[grid](
      *args, **kwargs, WHICH=which_val, zeroed_outputs=zeroed_fn
    )

    if zero_first:
      np.testing.assert_allclose(o0, x, err_msg="out0 should be zeroed (WHICH=1)")
      np.testing.assert_allclose(o1, 3 * x, err_msg="out1 should be dirty (WHICH=1)")
    else:
      np.testing.assert_allclose(o0, 3 * x, err_msg="out0 should be dirty (WHICH=0)")
      np.testing.assert_allclose(o1, x, err_msg="out1 should be zeroed (WHICH=0)")


if __name__ == "__main__":
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  absltest.main()
