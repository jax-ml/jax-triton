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

"""Contains fusion primitives and their lowering."""
import dataclasses
import functools
import os

from typing import Any, Tuple

import jax
from jax import core
from jax import lax
from jax import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax._src import util
from jax._src.lax.control_flow import for_loop
import jax.numpy as jnp
import jax_triton as jt
from jax_triton import pallas as pl

from jax_triton.experimental.fusion import jaxpr_rewriter
from oryx.experimental.matching import jax_rewrite
from oryx.experimental.matching import matcher


Eqn = jaxpr_rewriter.Eqn
Part = jaxpr_rewriter.Part

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

def lower_fused_jaxpr(jaxpr: core.Jaxpr, consts) -> core.Jaxpr:
  def _traceable(*args):
    return _eval_fused_jaxpr(jaxpr, consts, *args)
  in_avals = [v.aval for v in jaxpr.invars]
  lowered_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(_traceable), in_avals)
  return lowered_jaxpr, consts

lowering_rules = {}
def _eval_fused_jaxpr(jaxpr, consts, *args):
  env = {}

  def read_env(atom: core.Atom) -> Any:
    if isinstance(atom, core.Literal):
      return atom.val
    return env[atom]

  def write_env(var: core.Var, val: Any):
    env[var] = val

  map(write_env, jaxpr.invars, args)
  map(write_env, jaxpr.constvars, consts)

  for eqn in jaxpr.eqns:
    if eqn.primitive not in lowering_rules:
      raise NotImplementedError(eqn.primitive)
    rule = lowering_rules[eqn.primitive]
    invals = map(read_env, eqn.invars)
    outvals = rule(*invals, **eqn.params)
    if eqn.primitive.multiple_results:
      map(write_env, eqn.outvars, outvals)
    else:
      write_env(eqn.outvars[0], outvals)
  return map(read_env, jaxpr.outvars)


def _mul_lowering_rule(x, y):
  block_size = 512
  def _mul_scalar_kernel(x_ref, o_ref):
    pid = pl.program_id(0)
    idx = pid * block_size + jnp.arange(block_size)
    x = x_ref[idx]
    o_ref[idx] = x * y
  def _mul_kernel(x_ref, y_ref, o_ref):
    pid = pl.program_id(0)
    idx = pid * block_size + jnp.arange(block_size)
    x = x_ref[idx]
    y = y_ref[idx]
    o_ref[idx] = x + y
  num_blocks, remainder = divmod(x.size, block_size)
  num_blocks += bool(remainder)
  grid = lambda _: (num_blocks,)
  if y.shape == ():
    return pl.pallas_call(_mul_scalar_kernel, out_shape=jax.ShapeDtypeStruct(x.shape,
      x.dtype), grid=grid, num_warps=8,
        num_stages=3)(x)
  return pl.pallas_call(_mul_kernel, out_shape=jax.ShapeDtypeStruct(x.shape,
    x.dtype), grid=grid, num_warps=8,
      num_stages=3)(x, y)
lowering_rules[lax.mul_p] = _mul_lowering_rule

def _add_lowering_rule(x, y):
  block_size = 512
  def _add_scalar_kernel(x_ref, o_ref):
    pid = pl.program_id(0)
    idx = pid * block_size + jnp.arange(block_size)
    x = x_ref[idx]
    o_ref[idx] = x + y
  def _add_kernel(x_ref, y_ref, o_ref):
    pid = pl.program_id(0)
    idx = pid * block_size + jnp.arange(block_size)
    x = x_ref[idx]
    y = y_ref[idx]
    o_ref[idx] = x + y
  num_blocks, remainder = divmod(x.size, block_size)
  num_blocks += bool(remainder)
  grid = lambda _: (num_blocks,)
  if y.shape == ():
    return pl.pallas_call(_add_scalar_kernel, out_shape=jax.ShapeDtypeStruct(x.shape,
      x.dtype), grid=grid, num_warps=8,
        num_stages=3)(x)
  return pl.pallas_call(_add_kernel, out_shape=jax.ShapeDtypeStruct(x.shape,
    x.dtype), grid=grid, num_warps=8,
      num_stages=3)(x, y)
lowering_rules[lax.add_p] = _add_lowering_rule

def _sub_lowering_rule(x, y):
  return x - y
lowering_rules[lax.sub_p] = _sub_lowering_rule

def _div_lowering_rule(x, y):
  return x / y
lowering_rules[lax.div_p] = _div_lowering_rule

def _reduce_sum_lowering_rule(x, **params):
  return lax.reduce_sum_p.bind(x, **params)
lowering_rules[lax.reduce_sum_p] = _reduce_sum_lowering_rule

def _tanh_lowering_rule(x):
  block_size = 512
  def _tanh_kernel(x_ref, o_ref):
    pid = pl.program_id(0)
    idx = pid * block_size + jnp.arange(block_size)
    x = x_ref[idx]
    o_ref[idx] = jnp.tanh(x)
  num_blocks, remainder = divmod(x.size, block_size)
  num_blocks += bool(remainder)
  grid = lambda _: (num_blocks,)
  return pl.pallas_call(_tanh_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape,
    x.dtype), grid=grid,
      num_warps=4, num_stages=3)(x)
lowering_rules[lax.tanh_p] = _tanh_lowering_rule

def _logistic_lowering_rule(x):
  block_size = 512
  def _logistic_kernel(x_ref, o_ref):
    pid = pl.program_id(0)
    idx = pid * block_size + jnp.arange(block_size)
    x = x_ref[idx]
    o_ref[idx] = jax.nn.sigmoid(x)
  num_blocks, remainder = divmod(x.size, block_size)
  num_blocks += bool(remainder)
  grid = lambda _: (num_blocks,)
  return pl.pallas_call(_logistic_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape,
    x.dtype), grid=grid,
      num_warps=4, num_stages=3)(x)
lowering_rules[lax.logistic_p] = _logistic_lowering_rule

def _xla_call_lowering_rule(*args, call_jaxpr, **_):
  return _eval_fused_jaxpr(call_jaxpr, (), *args)
lowering_rules[xla.xla_call_p] = _xla_call_lowering_rule

elementwise_p = core.Primitive('elementwise')
elementwise_p.multiple_results = True

elementwise_p.def_abstract_eval(lambda *avals, **_: [avals[0]])

def _elementwise_lowering_rule(x, *, ops):
  block_size = 256
  def _elementwise_kernel(x_ref, o_ref):
    pid = pl.program_id(0)
    idx = pid * block_size + jnp.arange(block_size)
    x = x_ref[idx]
    args = (x,)
    for op in ops:
      args = op(*args)
    x, = args
    o_ref[idx] = x
  num_blocks, remainder = divmod(x.size, block_size)
  num_blocks += bool(remainder)
  grid = lambda _: (num_blocks,)
  return pl.pallas_call(_elementwise_kernel, out_shape=[jax.ShapeDtypeStruct(x.shape,
    x.dtype)], grid=grid, num_warps=8,
      num_stages=3)(x)
lowering_rules[elementwise_p] = _elementwise_lowering_rule


def make_elementwise(shape, dtype, *args):
  *args, ops = args
  return Part(0, shape, dtype, Eqn(elementwise_p, jax_rewrite.Params(ops=ops), list(args),
             [shape], [dtype]))

@dataclasses.dataclass(frozen=True)
class MatmulElementwise(jax_rewrite.JaxExpression):
  x: jax_rewrite.JaxExpression
  y: jax_rewrite.JaxExpression
  elem_ops: Tuple[core.Primitive]

  def match(self, expr, bindings, succeed):
    if not isinstance(expr, MatmulElementwise):
      return
    yield from matcher.matcher((self.elem_ops, self.x, self.y))((expr.elem_ops,
      expr.x, expr.y), bindings, succeed)

  def dtype(self):
    return self.x.dtype

  def shape(self):
    return (self.x.shape[0], self.y.shape[1])

  def evaluate(self, env):
    x = jax_rewrite.evaluate(self.x, env)
    y = jax_rewrite.evaluate(self.y, env)
    return matmul_elementwise_fusion_p.bind(
        x, y, eltwise_ops=self.elem_ops)

  def tree_map(self, fn):
    return MatmulElementwise(fn(self.x), fn(self.y), self.elem_ops)

  def tree_children(self):
    yield self.x
    yield self.y

  def __str__(self):
    return f"(fusion matmul_eltwise ({self.x}, {self.y}) {self.elem_ops})"

matmul_elementwise_fusion_p = core.Primitive("matmul_eltwise_fusion")

def _matmul_elementwise_fusion_impl(x, y, *args, **_):
  raise NotImplementedError
matmul_elementwise_fusion_p.def_impl(_matmul_elementwise_fusion_impl)

def _matmul_elementwise_fusion_abstract_eval(x, y, *args, **_):
  return core.ShapedArray((x.shape[0], y.shape[1]), x.dtype)
matmul_elementwise_fusion_p.def_abstract_eval(_matmul_elementwise_fusion_abstract_eval)

def _matmul_elementwise_lowering_rule(x, y, *args, left_ops, right_ops, out_ops,
                                      contract_dims):
  if len(args) == 1:
    bias, = args
  else:
    bias = None
  lhs_dim, rhs_dim = contract_dims
  M, N, K = x.shape[1 - lhs_dim], y.shape[1 - rhs_dim], x.shape[lhs_dim]
  assert x.shape[lhs_dim] == y.shape[rhs_dim]

  BLOCK_SIZE_M = min(256, M)
  BLOCK_SIZE_N = min(128, N)
  BLOCK_SIZE_K = min(32, K)
  GROUP_SIZE_M = 8


  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = (
      32, 64, 64, 4, 6)
  if "TRITON_CONFIG" in os.environ:
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = map(
        int, os.environ["TRITON_CONFIG"].split(","))

  @functools.partial(pl.pallas_call,
      out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
      grid=jt.cdiv(M, BLOCK_SIZE_M) * jt.cdiv(N, BLOCK_SIZE_N),
      num_warps=num_warps, num_stages=num_stages)
  def fused_matmul(x_ref, y_ref, *args):
    if len(args) == 2:
      bias_ref, o_ref = args
    else:
      bias_ref = None
      o_ref, = args
    pid = pl.program_id(axis=0)
    num_pid_m = M // BLOCK_SIZE_M
    num_pid_n = N // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = lax.div(pid, num_pid_in_group)
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = jnp.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + lax.rem(pid, group_size_m)
    pid_n = lax.div(lax.rem(pid, num_pid_in_group), group_size_m)
    idx_m = pid_m * BLOCK_SIZE_M + jnp.arange(BLOCK_SIZE_M)
    idx_n = pid_n * BLOCK_SIZE_N + jnp.arange(BLOCK_SIZE_N)
    idx_m = pl.max_contiguous(pl.multiple_of(idx_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    idx_n = pl.max_contiguous(pl.multiple_of(idx_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    def body(i, acc_ref):
      idx_k = i * BLOCK_SIZE_K + jnp.arange(BLOCK_SIZE_K)
      if lhs_dim == 1:
        x_idx = (
            jax.lax.broadcast_in_dim(idx_m, (BLOCK_SIZE_M, BLOCK_SIZE_K), (0,)),
            jax.lax.broadcast_in_dim(idx_k, (BLOCK_SIZE_M, BLOCK_SIZE_K), (1,)))
      else:
        x_idx = (
            jax.lax.broadcast_in_dim(idx_k, (BLOCK_SIZE_K, BLOCK_SIZE_M), (0,)),
            jax.lax.broadcast_in_dim(idx_m, (BLOCK_SIZE_K, BLOCK_SIZE_M), (1,)))
      if rhs_dim == 0:
        y_idx = (
            jax.lax.broadcast_in_dim(idx_k, (BLOCK_SIZE_K, BLOCK_SIZE_N), (0,)),
            jax.lax.broadcast_in_dim(idx_n, (BLOCK_SIZE_K, BLOCK_SIZE_N), (1,)))
      else:
        y_idx = (
            jax.lax.broadcast_in_dim(idx_n, (BLOCK_SIZE_N, BLOCK_SIZE_K), (0,)),
            jax.lax.broadcast_in_dim(idx_k, (BLOCK_SIZE_N, BLOCK_SIZE_K), (1,)))
      x_block, y_block = x_ref[x_idx], y_ref[y_idx]
      for eltwise_op in left_ops:
        x_block, = eltwise_op(x_block)
      for eltwise_op in right_ops:
        y_block, = eltwise_op(y_block)
      out = pl.dot(x_block, y_block, trans_a=lhs_dim == 0, trans_b=rhs_dim == 1,
                   allow_tf32=True)
      acc_ref[:, :] += out
    acc = jnp.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=jnp.float32)
    acc = for_loop.for_loop(K // BLOCK_SIZE_K, body, acc)
    if bias_ref is not None:
      b = bias_ref[idx_n]
      acc = acc + jax.lax.broadcast_in_dim(b, (BLOCK_SIZE_M, BLOCK_SIZE_N), (1,))
    for eltwise_op in out_ops:
      acc, = eltwise_op(acc)
    acc = acc.astype(x_ref.dtype)
    o_idx = (
        jax.lax.broadcast_in_dim(idx_m, (BLOCK_SIZE_M, BLOCK_SIZE_N), (0,)),
        jax.lax.broadcast_in_dim(idx_n, (BLOCK_SIZE_M, BLOCK_SIZE_N), (1,)),
        )
    o_ref[o_idx] = acc
  return fused_matmul(x, y, *args)
lowering_rules[matmul_elementwise_fusion_p] = _matmul_elementwise_lowering_rule

def _dot_general_lowering_rule(x, y, dimension_numbers, **_):
  contract_dims, batch_dims = dimension_numbers
  del batch_dims
  lhs_dim, rhs_dim = contract_dims[0][0], contract_dims[1][0]
  return _matmul_elementwise_lowering_rule(x, y, left_ops=[], right_ops=[],
                                           out_ops=[], contract_dims=(lhs_dim,
                                                                      rhs_dim))
lowering_rules[lax.dot_general_p] = _dot_general_lowering_rule

