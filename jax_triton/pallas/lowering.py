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

"""Module for lowering JAX primitives to Triton IR."""
import dataclasses

from typing import Any, Sequence

import jax
from jax import api_util
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax import lax
from jax._src import ad_util
from jax._src import util
from jax._src.lax.control_flow import for_loop
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax._src import state
from jax._src.state import primitives as sp
from jax._src.state import discharge
from jax._src.state import ShapedArrayRef
from jax_triton.triton_call import get_triton_python_ir
import jax.numpy as jnp
import triton
import triton.language as tl
import numpy as np
from triton.language import ir as tl_ir
import triton._C.libtriton.triton as _triton

from jax_triton.pallas import primitives

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

# # General lowering logic

@dataclasses.dataclass
class TritonModuleContext:
  name: str
  ir_context: tl_ir.context
  builder: tl_ir.builder
  module: tl_ir.module

@dataclasses.dataclass
class TritonLoweringRuleContext:
  context: TritonModuleContext
  avals_in: Any
  avals_out: Any

  def __post_init__(self):
    self.builder = self.context.builder

@dataclasses.dataclass
class TritonLoweringResult:
  """Keeps pybind11 objects alive."""
  ir_context: tl_ir.context
  module: tl_ir.module
  builder: tl_ir.builder

triton_lowering_rules = {}

def lower_jaxpr_to_triton_module(jaxpr: jax_core.Jaxpr, name: str) -> tl_ir.module:
  jaxpr, _ = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars), instantiate=True)
  ir_context = tl_ir.context()
  builder = tl_ir.builder(ir_context)
  module = tl_ir.module("", builder)
  ctx = TritonModuleContext(name, ir_context, builder, module)

  in_avals = [var.aval for var in jaxpr.invars]
  triton_types = [get_triton_python_ir(x) for x in in_avals]
  arg_types = [triton.compiler.str_to_ty(arg) for arg in triton_types]
  assert len(jaxpr.outvars) == 0
  ret_type = tl.void
  prototype = tl.function_type(ret_type, arg_types)
  out = prototype.to_ir(ctx.builder)
  fn = ctx.module.get_or_insert_function(name, out)
  args = []
  for i in range(len(in_avals)):
    fn.add_attr(i + 1, tl_ir.attribute(tl_ir.attribute_kind.aligned, 16))
    ptr = tl.tensor(fn.args[i], prototype.param_types[i])
    args.append(ptr)
  fn.set_is_kernel(True)
  insert_pt = ctx.builder.get_insert_block()
  entry = tl_ir.basic_block.create(ctx.builder.context, "entry", fn)
  ctx.builder.set_insert_block(entry)
  () = lower_jaxpr_to_triton_ir(ctx, jaxpr, *args)
  ctx.builder.ret_void()
  ctx.builder.set_insert_block(insert_pt)
  return TritonLoweringResult(ir_context, module, builder)

def lower_jaxpr_to_triton_ir(ctx: TritonModuleContext, jaxpr: jax_core.Jaxpr,
                             *args) -> tl_ir.module:

  env = {}
  def read_env(var: jax_core.Atom):
    if type(var) is jax_core.Literal:
      return tl.core._to_tensor(np.array(var.val).tolist(), builder=ctx.builder)
    return env[var]

  def write_env(var: jax_core.Var, val):
    env[var] = val

  map(write_env, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    if eqn.primitive not in triton_lowering_rules:
      raise NotImplementedError(eqn.primitive)
    rule = triton_lowering_rules[eqn.primitive]
    avals_in = [v.aval for v in eqn.invars]
    avals_out = [v.aval for v in eqn.outvars]
    rule_ctx = TritonLoweringRuleContext(ctx, avals_in, avals_out)
    outvals = rule(rule_ctx, *invals, **eqn.params)
    if eqn.primitive.multiple_results:
      map(write_env, eqn.outvars, outvals)
    else:
      write_env(eqn.outvars[0], outvals)
  return map(read_env, jaxpr.outvars)

# # Primitive lowering rules

# ## Programming model primitives

def _program_id_lowering_rule(ctx: TritonLoweringRuleContext, *, axis):
  return tl.program_id(axis, _builder=ctx.builder)
triton_lowering_rules[primitives.program_id_p] = _program_id_lowering_rule

# ## Atomic op primitives

_ATOMIC_OP_MAPPING = {
    primitives.AtomicOpType.XCHG: tl.core.atomic_xchg,
    primitives.AtomicOpType.ADD: tl.core.atomic_add,
    primitives.AtomicOpType.MAX: tl.core.atomic_max,
    primitives.AtomicOpType.MIN: tl.core.atomic_min,
    primitives.AtomicOpType.AND: tl.core.atomic_and,
    primitives.AtomicOpType.OR: tl.core.atomic_or,
    primitives.AtomicOpType.XOR: tl.core.atomic_xor,
}

def _atomic_lowering_rule(ctx: TritonLoweringRuleContext, ptr, value,
                          *non_slice_idx,
                          indexed_dims: Sequence[bool], masked: bool,
                          atomic_type: primitives.AtomicOpType):
  non_slice_idx, mask_rest = util.split_list(non_slice_idx, [sum(indexed_dims)])
  idx = _pack_indices(non_slice_idx, indexed_dims)
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  ptr = _offset_ptr(ptr, idx, avals_in[0].shape, avals_out[0].shape, ctx.builder)
  mask = None
  if masked:
    assert len(mask_rest) == 1
    mask, = mask_rest
  if atomic_type not in _ATOMIC_OP_MAPPING:
    raise NotImplementedError(atomic_type)
  op = _ATOMIC_OP_MAPPING[atomic_type]
  return op(ptr, value, mask=mask, _builder=ctx.builder)
triton_lowering_rules[primitives.atomic_rmw_p] = _atomic_lowering_rule

def _max_contiguous_lowering_rule(ctx: TritonLoweringRuleContext, x, *, values):
  values = [tl.constexpr(v) for v in values]
  return tl.max_contiguous(x, values, _builder=ctx.builder)
triton_lowering_rules[primitives.max_contiguous_p] = _max_contiguous_lowering_rule

def _multiple_of_lowering_rule(ctx: TritonLoweringRuleContext, x, *, values):
  values = [tl.constexpr(v) for v in values]
  return tl.multiple_of(x, values, _builder=ctx.builder)
triton_lowering_rules[primitives.multiple_of_p] = _multiple_of_lowering_rule

def _exp_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.exp(a, _builder=ctx.builder)
triton_lowering_rules[jax.lax.exp_p] = _exp_lowering_rule

def _logistic_lowering_rule(ctx: TritonLoweringRuleContext, a):
  one_= tl.core._to_tensor(1., ctx.builder)
  x = tl.exp(a.__neg__(_builder=ctx.builder), _builder=ctx.builder)
  x = x.__add__(one_, _builder=ctx.builder)
  x = one_.__truediv__(x, _builder=ctx.builder)
  return x
triton_lowering_rules[jax.lax.logistic_p] = _logistic_lowering_rule

def _sin_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.sin(a, _builder=ctx.builder)
triton_lowering_rules[jax.lax.sin_p] = _sin_lowering_rule

def _cos_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.cos(a, _builder=ctx.builder)
triton_lowering_rules[jax.lax.cos_p] = _cos_lowering_rule

def _mul_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__mul__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.mul_p] = _mul_lowering_rule

def _div_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  floating_dtypes = {tl.float16, tl.float32, tl.float64, tl.bfloat16}
  if a.dtype in floating_dtypes and b.dtype in floating_dtypes:
    return a.__truediv__(b, _builder=ctx.builder)
  return a.__floordiv__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.div_p] = _div_lowering_rule

def _iota_lowering_rule(ctx: TritonLoweringRuleContext, *, dtype, shape, dimension):
  if dimension != 0:
    raise NotImplementedError()
  return tl.arange(0, shape[0], _builder=ctx.builder)
triton_lowering_rules[jax.lax.iota_p] = _iota_lowering_rule

def _add_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__add__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.add_p] = _add_lowering_rule
triton_lowering_rules[ad_util.add_any_p] = _add_lowering_rule

def _integer_pow_lowering_rule(ctx: TritonLoweringRuleContext, a, *, y):
  if y == 2:
    return a.__mul__(a, _builder=ctx.builder)
  if y == 3:
    return a.__mul__(a.__mul__(a, _builder=ctx.builder), _builder=ctx.builder)
  return tl.libdevice.pow(a, y, _builder=ctx.builder)
triton_lowering_rules[jax.lax.integer_pow_p] = _integer_pow_lowering_rule

def _tanh_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.libdevice.tanh(a, _builder=ctx.builder)
triton_lowering_rules[jax.lax.tanh_p] = _tanh_lowering_rule

def _min_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  pred = a.__lt__(b, _builder=ctx.builder)
  return tl.semantic.where(pred, a, b, ctx.builder)
triton_lowering_rules[jax.lax.min_p] = _min_lowering_rule

def _convert_element_type_lowering_rule(ctx: TritonLoweringRuleContext, a, *,
                                        new_dtype, weak_type):
  if new_dtype == ctx.avals_in[0].dtype:
    return a
  if new_dtype == jnp.float32:
    new_dtype = tl.float32
  elif new_dtype == jnp.float16:
    new_dtype = tl.float16
  elif new_dtype == jnp.bfloat16:
    new_dtype = tl.bfloat16
  return tl.semantic.cast(a, new_dtype, ctx.builder)
triton_lowering_rules[jax.lax.convert_element_type_p] = _convert_element_type_lowering_rule

def max_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  pred = a.__gt__(b, _builder=ctx.builder)
  return tl.semantic.where(pred, a, b, ctx.builder)
triton_lowering_rules[jax.lax.max_p] = max_lowering_rule

def ge_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__ge__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.ge_p] = ge_lowering_rule

def select_n_lowering_rule(ctx: TritonLoweringRuleContext, pred, a, b):
  return tl.semantic.where(pred, a, b, ctx.builder)
triton_lowering_rules[jax.lax.select_n_p] = select_n_lowering_rule

def _rem_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__mod__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.rem_p] = _rem_lowering_rule

def _sub_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__sub__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.sub_p] = _sub_lowering_rule

def _lt_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__lt__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.lt_p] = _lt_lowering_rule

def _broadcast_in_dim_lowering_rule(ctx: TritonLoweringRuleContext, a, *, broadcast_dimensions, shape):
  # Add dummy dimensions
  if len(a.shape) != 1 or a.shape[0].value != 1:
    a_shape_iter = iter(a.shape)
    new_shape = [next(a_shape_iter) if i in broadcast_dimensions else 1
        for i in range(len(shape))]
    new_shape = tuple(tl.constexpr(v) for v in new_shape)
    a = tl.reshape(a, new_shape, _builder=ctx.builder)
  return tl.broadcast_to(a, list(shape), _builder=ctx.builder)
triton_lowering_rules[jax.lax.broadcast_in_dim_p] = _broadcast_in_dim_lowering_rule

def _squeeze_lowering_rule(ctx: TritonLoweringRuleContext, a, *, dimensions):
  shape = [tl.constexpr(s) for s in ctx.avals_out[0].shape]
  return tl.reshape(a, shape, _builder=ctx.builder)
triton_lowering_rules[jax.lax.squeeze_p] = _squeeze_lowering_rule

def _offset_ptr(ptr, indices, shape, out_shape, builder):
  strides = np.cumprod([1, *shape[::-1]])[:-1][::-1]
  offset = None
  for i, (idx, s, dim) in enumerate(zip(indices, strides, shape)):
    size = tl.core._to_tensor(int(s), builder)
    if isinstance(idx, slice):
      if idx != slice(None):
        raise NotImplementedError
      idx = tl.arange(0, dim, _builder=builder)
      if i > 0 and offset.shape != [1]:
        idx = tl.reshape(idx, (*([tl.constexpr(1)] *
          len(ptr.shape)), tl.constexpr(dim)), _builder=builder)
        dst_shape = [*offset.shape, tl.constexpr(1)]
        ret_ty = tl.block_type(offset.type.scalar, dst_shape)
        offset = tl.tensor(builder.create_reshape(offset.handle, dst_shape), ret_ty)
    idx = idx.__mul__(size, _builder=builder)
    if offset is None:
      offset = idx
    else:
      offset = offset.__add__(idx, _builder=builder)
  if offset is not None:
    ptr = ptr.__add__(offset, _builder=builder)
  if ptr.shape != [1] and ptr.shape == [1, *out_shape]:
    ptr = tl.reshape(ptr, map(tl.constexpr, out_shape), _builder=builder)
  return ptr

def _pack_indices(non_slice_idx, indexed_dims):
  non_slice_idx_iter = iter(non_slice_idx)
  return tuple(next(non_slice_idx_iter) if indexed else slice(None) for indexed
               in indexed_dims)

def _get_lowering_rule(ctx: TritonLoweringRuleContext, ptr, *non_slice_idx, indexed_dims):
  idx = _pack_indices(non_slice_idx, indexed_dims)
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  if not isinstance(ptr.type, tl.pointer_type):
    assert len(avals_in) == 1
    return ptr
  ptr = _offset_ptr(ptr, idx, avals_in[0].shape, avals_out[0].shape, ctx.builder)
  return tl.load(ptr, _builder=ctx.builder)
triton_lowering_rules[sp.get_p] = _get_lowering_rule

def _masked_load_lowering_rule(ctx: TritonLoweringRuleContext, ptr,
                               *non_slice_idx, indexed_dims, masked,
                               eviction_policy, cache_modifier, is_volatile):
  non_slice_idx, mask_other = util.split_list(non_slice_idx, [sum(indexed_dims)])
  idx = _pack_indices(non_slice_idx, indexed_dims)
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  if not isinstance(ptr.type, tl.pointer_type):
    assert len(avals_in) == 1
    return ptr
  ptr = _offset_ptr(ptr, idx, avals_in[0].shape, avals_out[0].shape, ctx.builder)
  mask, other = None, None
  if masked:
    assert 0 < len(mask_other) <= 2
    if len(mask_other) == 2:
      mask, other = mask_other
    elif len(mask_other) == 1:
      mask, = mask_other
  return tl.load(ptr, mask=mask, other=other, cache_modifier=cache_modifier,
                 volatile=is_volatile, eviction_policy=eviction_policy,
                 _builder=ctx.builder)
triton_lowering_rules[primitives.load_p] = _masked_load_lowering_rule

def _swap_lowering_rule(ctx: TritonLoweringRuleContext, ptr, value, *non_slice_idx, indexed_dims):
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  idx = _pack_indices(non_slice_idx, indexed_dims)
  ptr = _offset_ptr(ptr, idx, avals_in[0].shape, avals_out[0].shape, ctx.builder)
  mask = None
  old_value = tl.load(ptr, mask=mask, _builder=ctx.builder)
  tl.store(ptr, value, mask=mask, _builder=ctx.builder)
  return old_value
triton_lowering_rules[sp.swap_p] = _swap_lowering_rule

def _masked_swap_lowering_rule(ctx: TritonLoweringRuleContext, ptr, value,
                               *non_slice_idx, indexed_dims, masked,
                               eviction_policy):
  non_slice_idx, mask_other = util.split_list(non_slice_idx, [sum(indexed_dims)])
  idx = _pack_indices(non_slice_idx, indexed_dims)
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  ptr = _offset_ptr(ptr, idx, avals_in[0].shape, avals_out[0].shape, ctx.builder)
  mask = None
  if masked:
    assert len(mask_other) == 1
    mask, = mask_other
  return tl.store(ptr, value, mask=mask, eviction_policy=eviction_policy,
                  _builder=ctx.builder)
triton_lowering_rules[primitives.swap_p] = _masked_swap_lowering_rule

def _addupdate_lowering_rule(ctx: TritonLoweringRuleContext, ptr, value,
    *non_slice_idx, indexed_dims):
  idx = _pack_indices(non_slice_idx, indexed_dims)
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  mask = None
  ptr = _offset_ptr(ptr, idx, avals_in[0].shape, avals_out[0].shape, ctx.builder)
  old_value = tl.load(ptr, mask=mask, _builder=ctx.builder)
  tl.store(ptr, old_value.__add__(value, _builder=ctx.builder),
           mask=mask, _builder=ctx.builder)
  return []
triton_lowering_rules[sp.addupdate_p] = _addupdate_lowering_rule

def _dot_general_lowering(ctx: TritonLoweringRuleContext, a, b, *,
    dimension_numbers, precision, preferred_element_type):
  contract_dims, batch_dims = dimension_numbers
  assert batch_dims == ((), ())
  a_contract_dim, = contract_dims[0]
  b_contract_dim, = contract_dims[1]
  trans_a = a_contract_dim == 0
  trans_b = b_contract_dim == 1
  allow_tf32 = precision == lax.Precision.HIGH or precision == lax.Precision.DEFAULT
  return tl.dot(a, b, _builder=ctx.builder, trans_a=trans_a, trans_b=trans_b,
                allow_tf32=allow_tf32)
triton_lowering_rules[jax.lax.dot_general_p] = _dot_general_lowering

def _reduce_max_lowering(ctx: TritonLoweringRuleContext, a, *, axes):
  assert len(axes) == 1
  axis, = axes
  return tl.max(a, axis=axis, _builder=ctx.builder)
triton_lowering_rules[jax.lax.reduce_max_p] = _reduce_max_lowering

def _reduce_sum_lowering(ctx: TritonLoweringRuleContext, a, *, axes):
  assert len(axes) == 1
  axis, = axes
  return tl.sum(a, axis=axis, _builder=ctx.builder)
triton_lowering_rules[jax.lax.reduce_sum_p] = _reduce_sum_lowering

def _xla_call_lowering_rule(ctx: TritonLoweringRuleContext, *args, call_jaxpr, **_):
  return lower_jaxpr_to_triton_ir(ctx.context, call_jaxpr, *args)
triton_lowering_rules[xla.xla_call_p] = _xla_call_lowering_rule

def _for_lowering_rule(ctx: TritonLoweringRuleContext, *args, jaxpr,
    which_linear, nsteps, reverse, unroll):
  current_bb = ctx.builder.get_insert_block()
  loop_bb = _triton.ir.basic_block.create(ctx.builder.context, "loop", current_bb.parent)
  postloop_bb = _triton.ir.basic_block.create(ctx.builder.context, "postloop", current_bb.parent)
  orig_loop_counter = loop_counter = tl.core._to_tensor(0, ctx.builder)

  nsteps = tl.core._to_tensor(nsteps, ctx.builder)
  pred = loop_counter.__lt__(nsteps, _builder=ctx.builder)
  # If pred, we loop otherwise postloop
  ctx.builder.cond_br(pred.handle, loop_bb, postloop_bb)

  # Start populating the loop block
  ctx.builder.set_insert_block(loop_bb)

  instr = loop_bb.get_first_non_phi()
  ctx.builder.set_insert_point((loop_bb, instr))

  # Populate phi args for loop block (loop counter and values)
  should_discharge = [not isinstance(a, ShapedArrayRef) for a in ctx.avals_in]
  loop_counter = ctx.builder.create_phi(tl.int32.to_ir(ctx.builder), 2)
  ref_avals = [v.aval for v in jaxpr.invars][1:]
  read_only = [for_loop._is_read_only(eff) for eff in
               state.get_ref_state_effects(ref_avals, jaxpr.effects)]
  lowering_args = []
  for arg, sd, ro in zip(args, should_discharge, read_only):
    if not sd or ro:
      lowering_args.append(arg)
      continue
    lowering_args.append(tl.tensor(ctx.builder.create_phi(arg.type.to_ir(ctx.builder),
                                                          2), arg.type))

  loop_counter = loop_counter_phi = triton.language.tensor(loop_counter, tl.int32)
  # Partially discharge state from jaxpr for non-pointers
  discharged_jaxpr, () = discharge.discharge_state(jaxpr, (), should_discharge=[True, *should_discharge])
  # Inline discharged jaxpr into loop block
  ptr_args, _ = util.partition_list(should_discharge, lowering_args)
  new_args = lower_jaxpr_to_triton_ir(ctx.context, discharged_jaxpr,
                                      loop_counter, *lowering_args)
  new_args = util.merge_lists(should_discharge, ptr_args, new_args)
  # Increment loop and check pred + branch
  loop_counter = loop_counter.__add__(1, _builder=ctx.builder)
  pred = loop_counter.__lt__(nsteps, _builder=ctx.builder)
  ctx.builder.cond_br(pred.handle, loop_bb, postloop_bb)

  # Update phi nodes to point to outputs of loop block
  bb = loop_counter_phi.handle.get_parent()
  prec_bb, next_bb = bb.get_predecessors()
  loop_counter_phi.handle.add_incoming(orig_loop_counter.handle, prec_bb)
  loop_counter_phi.handle.add_incoming(loop_counter.handle, next_bb)
  for ro, d, old_arg, new_arg, phi_arg in zip(read_only, should_discharge, args,
                                              new_args, lowering_args):
    if not d or ro:
      continue
    phi_arg.handle.add_incoming(old_arg.handle, prec_bb)
    phi_arg.handle.add_incoming(new_arg.handle, next_bb)

  # Start populating post loop arrgs
  ctx.builder.set_insert_block(postloop_bb)

  # Set up phi nodes and populate
  post_args = []
  for ro, d, old_arg, new_arg in zip(read_only, should_discharge, args, new_args):
    if not d or ro:
      post_args.append(new_arg)
      continue
    phi_arg = tl.tensor(ctx.builder.create_phi(new_arg.type.to_ir(ctx.builder), 2),
                        new_arg.type)
    phi_arg.handle.add_incoming(old_arg.handle, prec_bb)
    phi_arg.handle.add_incoming(new_arg.handle, next_bb)
    post_args.append(phi_arg)
  return post_args
triton_lowering_rules[for_loop.for_p] = _for_lowering_rule
