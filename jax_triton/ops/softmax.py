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

"""Jax-triton wrapper for a Softmax kernel."""

import jax
import jax_triton as jt
import numpy as np

import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    # inputs
    input_ptr,
    # outputs
    probs_ptr,
    # dimension
    row_len,
    # block information
    # It is assumed that block_row >= row_len
    block_row: tl.constexpr
):
  row_index = tl.program_id(0)

  input_ptr += row_index * row_len
  probs_ptr += row_index * row_len

  row_tile = tl.arange(0, block_row)

  row = tl.load(
      input_ptr + row_tile,
      mask=row_tile < row_len,
      other=float("-inf")
  )

  row_max = tl.max(row, axis=0)
  numerator = tl.exp(row - row_max)
  denominator = tl.sum(numerator, axis=0)

  tl.store(
      probs_ptr + row_tile,
      numerator / denominator,
      mask=row_tile < row_len
  )


def softmax(x, *, axis: int = -1, num_warps: int = 4,
            num_stages: int = 8) -> jax.Array:
  """Implementation of Softmax.

  Args:
    x: input array
    axis: the axis along which to perform the computation
    num_warps: the number of warps to use for executing the Triton kernel
    num_stages: the number of stages to use when pipelining instructions

  Returns:
    The result of the softmax operation over the last axis of x.
  """
  axis = axis if axis >= 0 else len(x.shape) + axis
  if axis != len(x.shape) - 1:
    raise NotImplementedError(
        "reductions along non-trailing dimension unsupported")

  row_len = x.shape[-1]
  num_rows = int(np.prod(x.shape[:-1]))

  block_row = triton.next_power_of_2(row_len)

  metaparams = dict(
      block_row=block_row,
      num_stages=num_stages,
      num_warps=num_warps,
      row_len=row_len,
  )

  out_shape = [
      jax.ShapeDtypeStruct(x.shape, dtype=x.dtype),
  ]

  probs, = jt.triton_call(
      x,
      kernel=_softmax_kernel,
      call_name="triton_softmax",
      out_shape=out_shape,
      grid=lambda meta: (num_rows,),
      **metaparams,
  )

  return probs
