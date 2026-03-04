from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import config
from jax import random
import jax_triton as jt
import numpy as np
import triton

# import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
# from triton.language.extra import libdevice


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


# autotuner example isn't ported as Triton's autotuner depends on torch internally


class GluonTest(parameterized.TestCase):
  @parameterized.product(dtype=_JAX_DTYPES)
  def test_copy_scalar_kernel(self, dtype):
    def copy_scalar(input: jnp.ndarray) -> jnp.ndarray:
      assert input.size == 1 and input.ndim == 0
      # note, this also checks behaviour in the absense of metaparams args.
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


if __name__ == "__main__":
  absltest.main()
