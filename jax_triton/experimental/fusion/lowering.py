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

"""Contains lowering passes for jaxprs to pallas."""
import functools

from typing import Any, Dict

import jax
from jax import api_util
from jax import core
from jax import lax
from jax import linear_util as lu
from jax import tree_util
from jax._src import util
from jax._src import source_info_util
from jax.interpreters import partial_eval as pe

from jax_triton.experimental.fusion import fusion
from jax_triton.experimental.fusion import jaxpr_rewriter

from oryx.experimental.matching import jax_rewrite as jr
from oryx.experimental.matching import matcher


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

Var = matcher.Var
Dot = matcher.Dot
Segment = matcher.Segment
Eqn = jaxpr_rewriter.Eqn
Part = jaxpr_rewriter.Part

Sigmoid = lambda x: Eqn(lax.logistic_p, jr.Params(), [x], Dot, Dot)
Exp = lambda x: Eqn(lax.exp_p, jr.Params(), [x], Dot, Dot)
Add = lambda x, y: Eqn(lax.add_p, jr.Params(), [x, y], Dot, Dot)
Mul = lambda x, y: Eqn(lax.mul_p, jr.Params(), [x, y], Dot, Dot)
Sub = lambda x, y: Eqn(lax.sub_p, jr.Params(), [x, y], Dot, Dot)
Max = lambda x, y: Eqn(lax.max_p, jr.Params(), [x, y], Dot, Dot)
Ge = lambda x, y: Eqn(lax.ge_p, jr.Params(), [x, y], Dot, Dot)
IntegerPow = lambda x, y: Eqn(lax.integer_pow_p, jr.Params(y=y), [x], Dot, Dot)
Tanh = lambda x: Eqn(lax.tanh_p, jr.Params(), [x], Dot, Dot)

def _apply_all(rules):
  def rule(graph):
    done = False
    while not done:
      done = True
      rewritten = []
      for pattern, handler in rules:
        d = graph.rewrite_subgraph(pattern, handler)
        rewritten.append(d)
      done = not any(rewritten)
  return rule

Elementwise = lambda x, ops: Part(0, Eqn(fusion.elementwise_p,
  jr.Params(ops=ops), [x], Dot, Dot), Dot, Dot)

elementwise_of_elementwise = Eqn(
    fusion.elementwise_p,
    jr.Params(ops=Var('outer_ops')),
    [Segment('left'), Part(Var('idx'), Dot, Dot, Eqn(fusion.elementwise_p, jr.Params(ops=Var('inner_ops')),
      Var('inner_args'), Dot, Dot)), Segment('right')],
    Var('shape'), Var('dtype'))
def _elementwise_handler(idx, left, inner_args, right, inner_ops, outer_ops, shape,
    dtype):
  def wrapper_eltwise_op(*args):
    left_, inner, right = util.split_list(args, [len(left), len(inner_args)])
    for op in inner_ops:
      inner = op(*inner)
    args = [*left_, *inner, *right]
    for op in outer_ops:
      args = op(*args)
    return args
  return Eqn(
      fusion.elementwise_p, jr.Params(ops=[wrapper_eltwise_op]),
      [*left, *inner_args, *right], shape, dtype)

add_elemwise_pattern = Eqn(lax.add_p, jr.Params(),
    [Var('x'), Part(0, Dot, Dot, Eqn(fusion.elementwise_p,
    jr.Params(ops=Var('ops')), [Var('x')], Dot, Dot))], Var('shape'), Var('dtype'))
def _add_elemwise_handler(x, ops, shape, dtype):
  def _wrapper_op(*args):
    x_, = args
    for op in ops:
      args = op(*args)
    y, = args
    return [x_ + y]
  return Part(0, shape, dtype, Eqn(fusion.elementwise_p, jr.Params(ops=[_wrapper_op]),
             [x], [shape], [dtype]))

mul_elemwise_pattern = Eqn(lax.mul_p, jr.Params(),
    [Var('x'), Part(0, Dot, Dot, Eqn(fusion.elementwise_p,
    jr.Params(ops=Var('ops')), [Var('x')], Dot, Dot))], Var('shape'), Var('dtype'))
def _mul_elemwise_handler(x, ops, shape, dtype):
  def _wrapper_op(*args):
    x_, = args
    for op in ops:
      args = op(*args)
    y, = args
    return [x_ * y]
  return Part(0, shape, dtype, Eqn(fusion.elementwise_p, jr.Params(ops=[_wrapper_op]),
             [x], [shape], [dtype]))

fuse_elementwise = _apply_all([
  (elementwise_of_elementwise, _elementwise_handler),
  (add_elemwise_pattern, _add_elemwise_handler),
  (mul_elemwise_pattern, _mul_elemwise_handler),
  ])

dup_elementwise = Eqn(
    fusion.elementwise_p,
    jr.Params(ops=Var('ops')),
    [Segment('left'), Var('x'), Segment('middle'), Var('x'), Segment('right')],
    Var('shape'), Var('dtype'))
def _dup_elementwise_handler(left, x, middle, right, shape, dtype, ops):
  def wrapper_op(*args):
    left_, x, middle_, right_ = util.split_list(args, [len(left), 1, len(middle)])
    args = [*left_, *x, *middle_, *x, *right_]
    for op in ops:
      args = op(*args)
    return args
  return Eqn(
      fusion.elementwise_p, jr.Params(ops=[wrapper_op]),
      [*left, x, *middle, *right], shape, dtype)

dedup_elementwise = _apply_all([
  (dup_elementwise, _dup_elementwise_handler)
  ])

Matmul = lambda x, y, shape, dtype, ldim, rdim: Eqn(
    lax.dot_general_p, jr.Params(dimension_numbers=(((ldim,),
                                                     (rdim,)), ((), ())),
                                 precision=None,
                                 preferred_element_type=None), [x, y],
    shape, dtype)

transpose_matmul_pattern = Eqn(
    lax.transpose_p, jr.Params(permutation=(1, 0)),
    [Eqn(lax.dot_general_p, jr.Params(dimension_numbers=(((1,), (1,)), ((), ())),
                                      precision=None,
                                      preferred_element_type=None),
         [Var('x'), Var('y')], Dot, Dot)], Var('shape'), Var('dtype'))
def _transpose_matmul_handler(x, y, shape, dtype):
  return Eqn(
    lax.dot_general_p,
    jr.Params(dimension_numbers=(((1,), (1,)), ((), ())),
              precision=None,
              preferred_element_type=None), [y, x],
    shape, dtype)

ElementwiseFusedMatmul = lambda x, y, params, shape, dtype: Eqn(
    fusion.matmul_elementwise_fusion_p, params, [x, y], shape, dtype)

matmul_add_bias_pattern = Eqn(
    lax.add_p,
    jr.Params(),
    [
      Matmul(Var('x'), Var('y'), Var('shape'), Var('dtype'),
             Var('ldim'), Var('rdim')),
      Eqn(lax.broadcast_in_dim_p, jr.Params(broadcast_dimensions=(1,),
                                            shape=Dot),
          [Var('z')], Dot, Dot)
    ], Dot, Dot)
def _matmul_add_bias_handler(x, y, z, shape, dtype, ldim, rdim):
  return Eqn(fusion.matmul_elementwise_fusion_p,
             jr.Params(contract_dims=(ldim, rdim), left_ops=[], right_ops=[], out_ops=[]),
             [x, y, z], shape, dtype)

matmul_elementwise = Eqn(
    fusion.elementwise_p,
    jr.Params(ops=Var('ops')),
    [Matmul(Var('x'), Var('y'), Var('shape'), Var('dtype'), Var('ldim'),
    Var('rdim'))], Dot, Dot)
def _matmul_elementwise_handler(x, y, ops, shape, dtype, ldim, rdim):
  return Eqn(fusion.matmul_elementwise_fusion_p,
             jr.Params(left_ops=[], right_ops=[], out_ops=ops,
                       contract_dims=(ldim, rdim)),
             [x, y], shape, dtype)

left_elementwise_matmul = Matmul(
    Elementwise(Var('x'), Var('ops')), Var('y'), Var('shape'), Var('dtype'),
    Var('ldim'), Var('rdim'))

def _left_elementwise_matmul(x, y, ops, shape, dtype, ldim, rdim):
  return Eqn(fusion.matmul_elementwise_fusion_p, jr.Params(left_ops=ops, out_ops=ops),
             [x, y], shape, dtype)

right_elementwise_matmul = Matmul(
    Var('x'), Elementwise(Var('y'), Var('ops')), Var('shape'), Var('dtype'),
    Var('ldim'), Var('rdim'))

def _right_elementwise_matmul(x, y, ops, shape, dtype, ldim, rdim):
  return Eqn(fusion.matmul_elementwise_fusion_p, jr.Params(
    right_ops=ops, left_ops=[], out_ops=[]), [x, y], shape, dtype)

left_elementwise_fused_matmul = ElementwiseFusedMatmul(
    Elementwise(Var('x'), Var('ops')), Var('y'),
      Var('params'), Var('shape'), Var('dtype'))

def _left_elementwise_fused_matmul(x, y, ops, params, shape, dtype):
  return Eqn(fusion.matmul_elementwise_fusion_p,
             jr.Params(right_ops=params["right_ops"],
                       left_ops=[*ops, *params["left_ops"]],
                       out_ops=params["out_ops"]),
             [x, y], shape, dtype)

right_elementwise_fused_matmul = Eqn(
    fusion.matmul_elementwise_fusion_p,
    Var('params'),
    [Var('x'), Part(0, Dot, Dot, Eqn(
      fusion.elementwise_p,
      jr.Params(ops=Var('ops')), [Var('y')], Dot, Dot))],
    Var('shape'), Var('dtype'))

def _right_elementwise_fused_matmul(x, y, ops, params, shape, dtype):
  return Eqn(fusion.matmul_elementwise_fusion_p,
             jr.Params(right_ops=[*ops, *params["right_ops"]],
                       left_ops=params["left_ops"],
                       out_ops=params["out_ops"]),
             [x, y], shape, dtype)

out_elementwise_fused_matmul = Eqn(
    fusion.elementwise_p, jr.Params(ops=Var('ops')), [
      Eqn(fusion.matmul_elementwise_fusion_p,
          Var('params'), [Var('x'), Var('y'), Segment('bias')],
          Var('shape'), Var('dtype'))
      ], Dot, Dot)

def _out_elementwise_fused_matmul(x, y, bias, ops, params, shape, dtype):
  return Eqn(fusion.matmul_elementwise_fusion_p,
             jr.Params(out_ops=[*ops, *params["out_ops"]],
                       left_ops=params["left_ops"],
                       right_ops=params["right_ops"],
                       contract_dims=params["contract_dims"]),
             [x, y, *bias], shape, dtype)

fuse_matmul_elementwise = _apply_all([
  # (left_elementwise_matmul, _left_elementwise_matmul),
  # (right_elementwise_matmul, _right_elementwise_matmul),
  # (left_elementwise_fused_matmul, _left_elementwise_fused_matmul),
  # (right_elementwise_fused_matmul, _right_elementwise_fused_matmul),
  (transpose_matmul_pattern, _transpose_matmul_handler),
  (matmul_add_bias_pattern, _matmul_add_bias_handler),
  (matmul_elementwise, _matmul_elementwise_handler),
  (out_elementwise_fused_matmul, _out_elementwise_fused_matmul),
  ])

def _inline_calls(jaxpr: core.Jaxpr, consts) -> core.Jaxpr:
  _traceable = functools.partial(_eval_jaxpr_inline_calls, jaxpr, consts)
  in_avals = [v.aval for v in jaxpr.invars]
  inlined_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(_traceable), in_avals)
  return inlined_jaxpr, consts

elementwise_rules = {}

def _register_unary_elementwise_rule(prim):
  def rule(x, **params):
    def _op(*args):
      x, = args
      return [prim.bind(x, **params)]
    return fusion.elementwise_p.bind(x, ops=[_op])[0]
  elementwise_rules[prim] = rule

def _register_binary_elementwise_rule(prim):
  def rule(x, y, **params):
    if y.shape == ():
      def _op(*args):
        x, = args
        return [prim.bind(x, y.astype(x.dtype), **params)]
      return fusion.elementwise_p.bind(x, ops=[_op])[0]
    elif x.shape == ():
      def _op(*args):
        y, = args
        return [prim.bind(x.astype(y.dtype), y, **params)]
      return fusion.elementwise_p.bind(y, ops=[_op])[0]
    return prim.bind(x, y, **params)
  elementwise_rules[prim] = rule

_register_unary_elementwise_rule(lax.sin_p)
_register_unary_elementwise_rule(lax.cos_p)
_register_unary_elementwise_rule(lax.exp_p)
_register_unary_elementwise_rule(lax.logistic_p)
_register_unary_elementwise_rule(lax.integer_pow_p)
_register_unary_elementwise_rule(lax.tanh_p)
_register_binary_elementwise_rule(lax.mul_p)
_register_binary_elementwise_rule(lax.ge_p)
_register_binary_elementwise_rule(lax.max_p)
_register_binary_elementwise_rule(lax.min_p)
_register_binary_elementwise_rule(lax.add_p)
_register_binary_elementwise_rule(lax.sub_p)
_register_binary_elementwise_rule(lax.div_p)

def _select_n_elementwise_rule(pred, x, y):
  def _op(pred, x, y):
    return [lax.select_n_p.bind(pred, x, y)]
  return fusion.elementwise_p.bind(pred, x, y, ops=[_op])[0]
elementwise_rules[lax.select_n_p] = _select_n_elementwise_rule


def _eval_jaxpr_inline_calls(jaxpr: core.Jaxpr, consts, *args):
  def read(v: core.Atom) -> Any:
    return v.val if isinstance(v, core.Literal) else env[v]

  def write(v: Var, val: Any) -> None:
    env[v] = val

  env: Dict[Var, Any] = {}
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
    with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
      if isinstance(eqn.primitive, core.CallPrimitive):
        call_jaxpr = eqn.params["call_jaxpr"]
        ans = _eval_jaxpr_inline_calls(call_jaxpr, [], *map(read, eqn.invars))
      elif eqn.primitive in elementwise_rules:
        ans = elementwise_rules[eqn.primitive](*map(read, eqn.invars),
                                               **bind_params)
      else:
        ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  return map(read, jaxpr.outvars)

def lower_jaxpr(jaxpr: core.Jaxpr, consts, fuse: bool, debug: bool) -> core.Jaxpr:
  if debug:
    print("=========== Initial jaxpr ====================")
    print(jaxpr)
  jaxpr, consts = _inline_calls(jaxpr, consts)
  if debug:
    print("===== Inlining and detecting elementwise ops =========")
    print(jaxpr)
  graph = jaxpr_rewriter.JaxprGraph.from_jaxpr(jaxpr)
  if fuse:
    fuse_elementwise(graph)
    dedup_elementwise(graph)
    if debug:
      print("=========== Elementwise fusion ========================")
      print(graph.to_jaxpr())
    fuse_matmul_elementwise(graph)
    if debug:
      print("=========== Matmul elementwise fusion ========================")
      print(graph.to_jaxpr())
  jaxpr = graph.to_jaxpr()
  lowered_jaxpr, consts = fusion.lower_fused_jaxpr(jaxpr, consts)
  if debug:
    print("=========== Pallas lowering ========================")
    print(lowered_jaxpr)
  return lowered_jaxpr, consts

def jit(f, *, fuse: bool = True, debug: bool = False):
  @jax.jit
  def wrapped(*args, **kwargs):
    flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
    flat_fun, out_tree_thunk = api_util.flatten_fun(lu.wrap_init(f), in_tree)
    in_avals = [core.raise_to_shaped(core.get_aval(a)) for a in flat_args]
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    jaxpr, consts = lower_jaxpr(jaxpr, consts, fuse=fuse, debug=debug)
    out_vals = core.eval_jaxpr(jaxpr, consts, *flat_args)
    return tree_util.tree_unflatten(out_tree_thunk(), out_vals)
  return wrapped
