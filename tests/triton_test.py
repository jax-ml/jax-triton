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

from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector
    y_ptr,  # *Pointer* to second input vector
    length,  # Length of input and output vectors.
    output_ptr,  # *Pointer* to output vector
    BLOCK_SIZE: tl.constexpr,
):
  # There are multiple 'program's processing different data. We identify which program
  # we are here
  pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
  # This program will process inputs that are offset from the initial data.
  # for instance, if you had a vector of length 256 and block_size of 64, the programs
  # would each access the elements [0:64, 64:128, 128:192, 192:256].
  # Note that offsets is a list of pointers
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  # Create a mask to guard memory operations against out-of-bounds accesses
  mask = offsets < length
  # Load x and y from DRAM, masking out any extra elements in case the input is not a
  # multiple of the block size
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  # Write x + y back to DRAM
  tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def tanh_kernel(
    x_ptr,  # *Pointer* to first input vector
    length,  # Length of input and output vectors.
    output_ptr,  # *Pointer* to output vector
    BLOCK_SIZE: tl.constexpr,
):
  # There are multiple 'program's processing different data. We identify which program
  # we are here
  pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
  # This program will process inputs that are offset from the initial data.
  # for instance, if you had a vector of length 256 and block_size of 64, the programs
  # would each access the elements [0:64, 64:128, 128:192, 192:256].
  # Note that offsets is a list of pointers
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  # Create a mask to guard memory operations against out-of-bounds accesses
  mask = offsets < length
  # Load x and y from DRAM, masking out any extra elements in case the input is not a
  # multiple of the block size
  x = tl.load(x_ptr + offsets, mask=mask)
  output = libdevice.tanh(x)
  # Write x + y back to DRAM
  tl.store(output_ptr + offsets, output, mask=mask)


class TritonTest(absltest.TestCase):

  def test_add_kernel(self):

    def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
      out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
      grid = lambda meta: (triton.cdiv(x.size, meta['BLOCK_SIZE']),)
      return jt.triton_call(
          x,
          y,
          x.size,
          kernel=add_kernel,
          out_shape=out_shape,
          grid=grid,
          BLOCK_SIZE=8,
      )

    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.arange(8, dtype=jnp.float32)
    np.testing.assert_allclose(add(x, y), x + y)

  def test_tanh_kernel(self):

    def tanh(x: jnp.ndarray) -> jnp.ndarray:
      out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
      grid = lambda meta: (triton.cdiv(x.size, meta['BLOCK_SIZE']),)
      return jt.triton_call(
          x,
          x.size,
          kernel=tanh_kernel,
          out_shape=out_shape,
          grid=grid,
          BLOCK_SIZE=8,
      )

    x = jnp.arange(8, dtype=jnp.float32)
    np.testing.assert_allclose(tanh(x), np.tanh(x))


if __name__ == '__main__':
  absltest.main()
