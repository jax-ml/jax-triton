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

"""Addition example."""
import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    block_size: tl.constexpr,
):
  """Adds two vectors."""
  pid = tl.program_id(axis=0)
  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)
  mask = offsets < 8
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)

def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
  block_size = 8
  grid = (triton.cdiv(x.size, block_size),)
  return jt.triton_call(
      x,
      y,
      kernel=add_kernel,
      out_shape=out_shape,
      grid=grid,
      block_size=block_size)


def main(unused_argv):
  x_val = jnp.arange(8)
  y_val = jnp.arange(8, 16)
  print(add(x_val, y_val))
  print(jax.jit(add)(x_val, y_val))

if __name__ == "__main__":
  from absl import app
  app.run(main)
