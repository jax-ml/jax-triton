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

"""Module containing fused matmuls."""
import functools

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import for_loop

import jax_triton as jt
from jax_triton import pallas as pl

def _compute_bound_configs():
  yield from [
    dict(bm=128, bn=256, bk=32, compiler_params=dict(triton=dict(num_stages=3, num_warps=8))),
    dict(bm=256, bn=128, bk=32, compiler_params=dict(triton=dict(num_stages=3, num_warps=8))),
    dict(bm=256, bn=64 , bk=32, compiler_params=dict(triton=dict(num_stages=4, num_warps=4))),
    dict(bm=64,  bn=256, bk=32, compiler_params=dict(triton=dict(num_stages=4, num_warps=4))),
    dict(bm=128, bn=128, bk=32, compiler_params=dict(triton=dict(num_stages=4, num_warps=4))),
    dict(bm=128, bn=64,  bk=32, compiler_params=dict(triton=dict(num_stages=4, num_warps=4))),
    dict(bm=64,  bn=128, bk=32, compiler_params=dict(triton=dict(num_stages=4, num_warps=4))),
    dict(bm=128, bn=32,  bk=32, compiler_params=dict(triton=dict(num_stages=4, num_warps=4))),
    dict(bm=64,  bn=32,  bk=32, compiler_params=dict(triton=dict(num_stages=5, num_warps=2))),
  ]

@functools.partial(jax.jit, static_argnames=["interpret", "debug"])
def matmul(x, y, *, interpret=False, debug=False):
  # TODO(sharadmv): make this implementation better
  # 1. reordered programs for better L2
  # 2. split K
  # 3. masking
  m, n, k = x.shape[0], y.shape[1], x.shape[1]

  @functools.partial(
      pl.pallas_call, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      autotuning_configs=[
        pl.KernelConfig(
          name=f"bm={config['bm']}_bn={config['bn']}_bk={config['bk']}",
          meta=dict(bk=config["bk"]),
          in_specs=[
            pl.BlockSpec(lambda i, _: (i, 0), (config["bm"], x.shape[1])),
            pl.BlockSpec(lambda _, j: (0, j), (y.shape[0], config["bn"]))
          ],
          out_specs=pl.BlockSpec(lambda i, j: (i, j), (config["bm"],
                                                       config["bn"])),
          grid=(jt.cdiv(m, config["bm"]), jt.cdiv(n, config["bn"])),
          compiler_params=config["compiler_params"],
        )
        for config in _compute_bound_configs()
      ]
  )
  def matmul_kernel(x_ref, y_ref, o_ref, *, bk: int):
    acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)
    def body(i, acc_ref):
      x_block = pl.load(x_ref, (slice(None), pl.ds(i * bk, bk)))
      y_block = pl.load(y_ref, (pl.ds(i * bk, bk), slice(None)))
      acc_ref[:, :] += jnp.dot(x_block, y_block)
    acc = for_loop(k // bk, body, acc).astype(o_ref.dtype)
    o_ref[:, :] = acc
  return matmul_kernel(x, y)

@jax.jit
def matmul_reference(x, y):
  return jnp.dot(x, y)
