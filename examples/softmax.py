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

"""Softmax example."""
import math

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

next_pow2 = lambda x: int(math.pow(2, math.ceil(math.log(x, 2))))


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride: tl.constexpr, output_row_stride: tl.constexpr, n_cols:
    tl.constexpr, block_size: tl.constexpr
):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, block_size)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x: jnp.ndarray) -> jnp.ndarray:
  out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
  block_size = next_pow2(x.shape[1])
  strides = jt.strides_from_shape(x.shape)
  return jt.triton_call(
      x,
      kernel=softmax_kernel,
      out_shape=out_shape,
      input_row_stride=strides[0],
      output_row_stride=strides[0],
      n_cols=x.shape[1],
      grid=x.shape[0],
      block_size=block_size)


def main(unused_argv):
  x_val = jnp.ones((8, 5), dtype="float32")
  print(softmax(x_val).block_until_ready())
  print(jax.jit(softmax)(x_val).block_until_ready())

if __name__ == "__main__":
  from absl import app
  app.run(main)
