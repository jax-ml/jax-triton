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

import functools
import timeit


from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

import jax_triton as jt
import tensorflow_probability.substrates.jax as tfp
from jax_triton import pallas as pl

tfd = tfp.distributions

def sdd_kernel(x_ref, indices_ref, blocks_per_row_ref, y_ref, o_ref, *, bm, bn):
  pid_m = pl.program_id(axis=0)
  pid_n = pl.program_id(axis=1)

  # Pseudocode:
  # for i in range(nrows):  # grid m
  #   num_blocks_in_row = self.blocks_per_row[i]
  #   for j in range(ncols):  # grid n
  #     acc = jnp.zeros((n, other.shape[1]))
  #     for k in range(num_blocks_in_row):
  #       jj, sparse_idx = indices[i, k]
  #       chunk = lax.dynamic_slice(other, [jj * m, 0], (m, other.shape[1]))
  #       block = self.blocks[sparse_idx]
  #       acc += block.dot(chunk)
  #       num_dots += 1
  #     out = out.at[i * n:(i + 1) * n, :].set(acc)

  num_blocks = blocks_per_row_ref[pid_m]
  acc = jnp.zeros((bm, bn), dtype=jnp.float32)
  def body(k, acc):
    jj = indices_ref[pid_m, k, 0]
    sparse_idx = indices_ref[pid_m, k, 1]
    block = pl.load(x_ref, (sparse_idx, pl.dslice(None), pl.dslice(None)))
    chunk = pl.load(y_ref, (pl.dslice(jj * bm, bm), pl.dslice(pid_n * bn, bn)))
    return acc + pl.dot(block, chunk)
  acc = lax.fori_loop(0, num_blocks, body, acc).astype(o_ref.dtype)
  pl.store(o_ref, (pl.dslice(bm * pid_m, bm), pl.dslice(bn * pid_n, bn)), acc)


@jax.tree_util.register_pytree_node_class
class BlockELL:
  blocks: jnp.array  # float32[n_rows, n_blocks, *block_size]
  blocks_per_row: jnp.array  # int32[n_rows, n_blocks]
  indices: jnp.array  # int32[n_rows, max_num_blocks_per_row, 2]
  shape: Tuple[int, int] # (n_rows * block_size[0], n_cols * block_size[1])

  ndim: int = property(lambda self: len(self.shape))
  num_blocks = property(lambda self: self.blocks.shape[0])
  block_size = property(lambda self: self.blocks.shape[1:])
  dtype = property(lambda self: self.blocks.dtype)

  def __init__(self, blocks, blocks_per_row, indices, *, shape):
    self.blocks = blocks
    self.blocks_per_row = blocks_per_row
    self.indices = indices
    self.shape = shape

  def tree_flatten(self):
    return (self.blocks, self.blocks_per_row, self.indices), (self.shape,)

  @classmethod
  def tree_unflatten(cls, data, xs):
    blocks, blocks_per_row, indices = xs
    shape, = data
    return BlockELL(blocks, blocks_per_row, indices, shape=shape)

  def _validate(self):
    nblocks, n, m = self.blocks.shape
    nrows = self.blocks_per_row.shape[0]
    assert self.indices.shape[0] == nrows
    assert len(self.shape) == 2
    assert self.shape[0] == n * nrows
    assert self.shape[1] % m == 0

  @jax.jit
  def todense(self) -> jnp.ndarray:
    self._validate()
    _, n, m = self.blocks.shape
    nrows = self.shape[0] // n
    out = jnp.zeros(self.shape, self.dtype)
    def i_body(i, out):
      num_blocks_in_row = self.blocks_per_row[i]
      def j_body(j, out):
        jj, sparse_idx = self.indices[i, j]
        out = lax.dynamic_update_slice(out, self.blocks[sparse_idx], (i * n, jj * m))
        return out
      return lax.fori_loop(0, num_blocks_in_row, j_body, out)
    return lax.fori_loop(0, nrows, i_body, out)

  @jax.jit
  def __matmul__(self, other):
    assert isinstance(other, jnp.ndarray)
    self._validate()
    assert self.ndim == other.ndim == 2
    assert self.shape[1] == other.shape[0]
    out = jnp.zeros((self.shape[0], other.shape[1]),
                    dtype=jnp.result_type(self.dtype, other.dtype))
    _, n, m = self.blocks.shape
    nrows = self.shape[0] // n
    def i_body(i):
      num_blocks_in_row = self.blocks_per_row[i]
      acc = jnp.zeros((n, other.shape[1]), dtype=jnp.float32)
      def k_body(k, acc):
        jj, sparse_idx = self.indices[i, k]
        chunk = lax.dynamic_slice(other, [jj * m, 0], (m, other.shape[1]))
        block = self.blocks[sparse_idx]
        return acc + block.dot(chunk)
      acc = lax.fori_loop(0, num_blocks_in_row, k_body, acc).astype(out.dtype)
      return acc
    accs = jax.vmap(i_body)(jnp.arange(nrows))
    return accs.reshape((self.shape[0], other.shape[1]))

def sample_sparse_matrix(key, m, n, bm, bn, *, sparse_prob=0.2,
                         dtype=jnp.float32) -> BlockELL:
  k1, k2, k3 = random.split(key, 3)
  num_rows = m // bm
  num_cols = n // bn
  blocks_per_row = tfd.Binomial(num_cols, probs=sparse_prob).sample(
      seed=k1, sample_shape=[num_rows]).astype(jnp.int32)
  num_blocks = blocks_per_row.sum()
  indices = []
  block_index = 0
  max_num_blocks = blocks_per_row.max(axis=0)
  for i, k in zip(range(num_rows), random.split(k2, num_rows)):
    row = []
    num_blocks_in_row = blocks_per_row[i]
    block_indices = jnp.sort(random.permutation(k, jnp.arange(num_cols))[:max_num_blocks])
    for j, b in zip(range(max_num_blocks), block_indices):
      if j < num_blocks_in_row:
        index = [b, block_index]
        block_index += 1
      else:
        index = [0, 0]
      row.append(index)
    indices.append(row)
  indices = jnp.array(indices)
  blocks = random.normal(k3, (num_blocks, bm, bn), dtype=dtype)
  return BlockELL(blocks, blocks_per_row, indices, shape=(m, n))

@functools.partial(jax.jit, static_argnames=["bn", "num_warps", "num_stages",
                                             "debug"])
def sdd_matmul(x_ell, y, num_warps: int = 8, num_stages: int = 3, bn: int = 64,
               debug: bool = False):
  m, n = x_ell.shape[0], y.shape[1]
  _, bm, _ = x_ell.blocks.shape
  grid = (jt.cdiv(m, bm), jt.cdiv(n, bn))

  kernel = functools.partial(sdd_kernel, bm=bm, bn=bn)
  out_shape = jax.ShapeDtypeStruct(shape=(m, n), dtype=x.dtype)
  return pl.pallas_call(kernel, num_warps=num_warps, num_stages=num_stages,
                        grid=grid, out_shape=out_shape,
                        debug=debug)(x_ell.blocks, x_ell.indices,
                                     x_ell.blocks_per_row, y)

if __name__ == "__main__":
  k1, k2 = random.split(random.PRNGKey(0))
  dtype = jnp.float16
  m, k, n = 4096, 4096, 4096
  bm, bk, bn = 32, 32, 256
  sparse_prob = 0.1
  x = sample_sparse_matrix(k1, m, k, bm, bk, sparse_prob=sparse_prob, dtype=dtype)
  print(f"Sparsity: {x.num_blocks} / {m // bm * k // bk}")
  x_dense = x.todense()
  y = random.normal(k2, (k, n), dtype=dtype)
  sdd_matmul(x, y, bn=bn, debug=True).block_until_ready()
  sparse_matmul = jax.jit(functools.partial(sdd_matmul, bn=bn))
  dense_matmul = jax.jit(jnp.matmul)
  out = sparse_matmul(x, y)
  out_hlo = (x @ y).block_until_ready()
  out_ref = jnp.matmul(x_dense, y)
  np.testing.assert_allclose(out, out_ref, atol=0.04, rtol=0.04)
  np.testing.assert_allclose(out_hlo, out_ref, atol=0.04, rtol=0.04)

  print("Starting benchmark...")
  n_trials = 10000
  duration = timeit.timeit(lambda: dense_matmul(x_dense, y).block_until_ready(),
                           number=n_trials)
  print(f"Dense Matmul: {duration / n_trials * 1000:.2f}ms")
  duration = timeit.timeit(lambda: sparse_matmul(x, y).block_until_ready(),
                           number=n_trials)
  print(f"Triton Blocksparse Matmul: {duration / n_trials * 1000:.2f}ms")
  n_trials = 20  # So slow!
  duration = timeit.timeit(lambda: (x @ y).block_until_ready(),
                           number=n_trials)
  print(f"HLO Blocksparse Matmul: {duration / n_trials * 1000:.2f}ms")
