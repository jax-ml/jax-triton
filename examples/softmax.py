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

"""Softmax example."""
import math

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

next_pow2 = lambda x: int(math.pow(2, math.ceil(math.log(x, 2))))


@triton.jit
def softmax_kernel(input_ptr, output_ptr, block_size: tl.constexpr,
                   size: tl.constexpr):
  """Computes the softmax of a vector."""
  input_row_stride = 1
  output_row_stride = 1
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * input_row_stride
  col_offsets = tl.arange(0, block_size)
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < size, other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * output_row_stride
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < size)


def softmax(x: jnp.ndarray) -> jnp.ndarray:
  out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
  size = x.shape[0]
  block_size = next_pow2(size)
  grid = (triton.cdiv(x.size, block_size),)
  return jt.triton_call(
      x,
      kernel=softmax_kernel,
      out_shape=out_shape,
      size=x.shape[0],
      grid=grid,
      block_size=next_pow2(x.shape[0]))


if __name__ == "__main__":
  x_val = jnp.ones(1024, dtype="float32")
  print(softmax(x_val).block_until_ready())
  print(jax.jit(softmax)(x_val).block_until_ready())
