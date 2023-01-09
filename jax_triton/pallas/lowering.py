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
import functools
import operator

from typing import Any, Optional, Tuple, Sequence

import jax
from jax import api_util
from jax import linear_util as lu
from jax import tree_util
from jax import lax
from jax._src import ad_util
from jax._src import util
from jax._src.lax.control_flow import for_loop
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax._src import core as jax_core
from jax._src import state
from jax._src.state import primitives as sp
from jax._src.state import discharge
from jax._src.state import ShapedArrayRef
from jax._src.util import partition_list, merge_lists
from jax_triton.triton_call import get_triton_python_ir
import jax.numpy as jnp
import triton
import triton.language as tl
import numpy as np
from triton.language import ir as tl_ir
import triton._C.libtriton.triton as _triton

import jax_triton as jt
from jax_triton.pallas import primitives
from jax_triton.pallas import core as pallas_core

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip
partial = functools.partial

GridSpec = pallas_core.GridSpec
BlockMapping = pallas_core.BlockMapping

# # General lowering logic

@dataclasses.dataclass
class TritonModuleContext:
  name: str
  ir_context: tl_ir.context
  builder: tl_ir.builder
  module: tl_ir.module
  grid_spec: GridSpec
  program_ids: Sequence[tl.tensor]

@dataclasses.dataclass
class BlockInfo:
  full_shape: Tuple[int, ...]
  start_indices: Sequence[Any]
  block_shape: Tuple[int, ...]

@dataclasses.dataclass
class TritonLoweringRuleContext:
  context: TritonModuleContext
  avals_in: Any
  avals_out: Any
  block_infos: Sequence[Optional[BlockInfo]]

  def __post_init__(self):
    self.builder = self.context.builder

@dataclasses.dataclass
class TritonLoweringResult:
  """Keeps pybind11 objects alive."""
  ir_context: tl_ir.context
  module: tl_ir.module
  builder: tl_ir.builder

def _eval_index_map(ctx: TritonModuleContext, idx, block_mapping: Optional[BlockMapping]):
  if block_mapping is None:
    return None
  block_indices = tuple(lower_jaxpr_to_triton_ir(
    ctx, block_mapping.index_map_jaxpr.jaxpr, None, *idx))
  return tuple(
      i if b is pallas_core.mapped else i.__mul__(b, _builder=ctx.builder)
      for i, b in zip(block_indices, block_mapping.block_shape))

triton_lowering_rules = {}

def lower_jaxpr_to_triton_module(jaxpr: jax_core.Jaxpr, in_shapes, grid_spec: GridSpec,
                                 name: str) -> tl_ir.module:
  jaxpr, _ = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars), instantiate=True)
  ir_context = tl_ir.context()
  ir_context.load_triton()
  builder = tl_ir.builder(ir_context)
  module = builder.create_module()
  in_avals = [var.aval for var in jaxpr.invars]
  triton_types = [get_triton_python_ir(x) for x in in_avals]
  arg_types = [triton.compiler.str_to_ty(arg) for arg in triton_types]
  assert len(jaxpr.outvars) == 0
  prototype = tl.function_type([], arg_types)
  out = prototype.to_ir(builder)
  fn = builder.get_or_insert_function(module, name, out, "public")
  module.push_back(fn)
  entry = fn.add_entry_block()
  args = []
  for i in range(len(in_avals)):
    fn.set_arg_attr(i, "tt.divisibility", 16)
    ptr = tl.tensor(fn.args(i), prototype.param_types[i])
    args.append(ptr)
  insert_pt = builder.get_insertion_block()
  builder.set_insertion_point_to_start(entry)
  program_ids = [tl.program_id(axis=i, _builder=builder)
                 for i in range(len(grid_spec.grid))]
  local_program_ids = [pid for i, pid in enumerate(program_ids)
                       if i not in grid_spec.mapped_dims]
  ctx = TritonModuleContext(name, ir_context, builder, module, grid_spec,
                            local_program_ids)
  start_indices = map(partial(_eval_index_map, ctx, program_ids),
                      grid_spec.block_mappings)
  block_infos = [BlockInfo(shape,
                 start_idx, block_mapping.block_shape)
                 if block_mapping is not None else None
                 for shape, block_mapping, start_idx in
                 zip(in_shapes, grid_spec.block_mappings, start_indices)]
  () = lower_jaxpr_to_triton_ir(ctx, jaxpr, block_infos, *args)
  module.context = ir_context
  ctx.builder.ret([])
  return TritonLoweringResult(ir_context, module, builder)

def lower_jaxpr_to_triton_ir(ctx: TritonModuleContext, jaxpr: jax_core.Jaxpr,
                             block_infos: Optional[Sequence[Optional[BlockInfo]]],
                             *args) -> tl_ir.module:
  env = {}
  block_info_env = {}
  def read_env(var: jax_core.Atom):
    if type(var) is jax_core.Literal:
      return tl.core._to_tensor(np.array(var.val).tolist(), builder=ctx.builder)
    return env[var]

  def read_block_info_env(var: jax_core.Atom):
    if type(var) is jax_core.Literal:
      return None
    return block_info_env.get(var, None)

  def write_env(var: jax_core.Var, val):
    env[var] = val

  if block_infos is None:
    block_infos = [None] * len(jaxpr.invars)
  for invar, block_info in zip(jaxpr.invars, block_infos):
    block_info_env[invar] = block_info

  map(write_env, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    if eqn.primitive not in triton_lowering_rules:
      raise NotImplementedError(eqn.primitive)
    rule = triton_lowering_rules[eqn.primitive]
    avals_in = [v.aval for v in eqn.invars]
    avals_out = [v.aval for v in eqn.outvars]
    eqn_block_infos = map(read_block_info_env, eqn.invars)
    rule_ctx = TritonLoweringRuleContext(
        ctx, avals_in, avals_out, eqn_block_infos)
    outvals = rule(rule_ctx, *invals, **eqn.params)
    if eqn.primitive.multiple_results:
      map(write_env, eqn.outvars, outvals)
    else:
      write_env(eqn.outvars[0], outvals)
  return map(read_env, jaxpr.outvars)

# # Primitive lowering rules

# ## Programming model primitives

def _program_id_lowering_rule(ctx: TritonLoweringRuleContext, *, axis):
  return ctx.context.program_ids[axis]
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
                          *args,
                          args_tree, masked: bool,
                          atomic_type: primitives.AtomicOpType):
  ref_block_info, *_ = ctx.block_infos
  idx, *mask_rest = tree_util.tree_unflatten(args_tree, args)
  avals_in = ctx.avals_in
  idx_avals, *_ = tree_util.tree_unflatten(args_tree, avals_in[2:])
  is_scalar = [hasattr(a, "shape") and a.shape == () for a in
               idx_avals.indices]
  ptr = _offset_ptr(ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder, is_scalar)
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

def _log_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.log(a, _builder=ctx.builder)
triton_lowering_rules[jax.lax.log_p] = _log_lowering_rule

def _log1p_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.libdevice.log1p(a, _builder=ctx.builder)
triton_lowering_rules[jax.lax.log1p_p] = _log1p_lowering_rule

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
  if y == -2:
    return tl.libdevice.rsqrt(a, _builder=ctx.builder)
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
  return tl.semantic.where(pred, b, a, ctx.builder)
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

def _sqrt_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.sqrt(a, _builder=ctx.builder)
triton_lowering_rules[jax.lax.sqrt_p] = _sqrt_lowering_rule

def _neg_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return a.__neg__(_builder=ctx.builder)
triton_lowering_rules[jax.lax.neg_p] = _neg_lowering_rule

def _broadcast_in_dim_lowering_rule(ctx: TritonLoweringRuleContext, a, *, broadcast_dimensions, shape):
  # There are no scalars in Triton so we need to handle the case where we have a
  # [1]-shaped value that is logically a JAX scalar.
  is_scalar = ctx.avals_in[0].shape == ()
  # Add dummy dimensions
  if not is_scalar:
    expand_dims = [i for i in range(len(shape)) if i not in broadcast_dimensions]
    for dim in expand_dims:
      a = tl.semantic.expand_dims(a, dim, ctx.builder)
  return tl.core.broadcast_to(a, shape, _builder=ctx.builder)
triton_lowering_rules[jax.lax.broadcast_in_dim_p] = _broadcast_in_dim_lowering_rule

def _squeeze_lowering_rule(ctx: TritonLoweringRuleContext, a, *, dimensions):
  shape = [tl.constexpr(s) for s in ctx.avals_out[0].shape]
  return tl.reshape(a, shape, _builder=ctx.builder)
triton_lowering_rules[jax.lax.squeeze_p] = _squeeze_lowering_rule

def _offset_ptr(ptr, block_info: Optional[BlockInfo], idx: primitives.NDIndexer, shape, builder,
                is_scalar):
  if block_info is None:
    full_shape = shape
    num_mapped_dims = 0
    block_shape = shape
  else:
    full_shape = block_info.full_shape.shape
    num_mapped_dims = sum(b is pallas_core.mapped for b in block_info.block_shape)
    block_shape = block_info.block_shape
  strides = jt.strides_from_shape(full_shape)
  indexer_shape = idx.get_indexer_shape()
  indices = idx.indices
  other_shape = indexer_shape[len(idx.int_indexer_shape):]
  bcast_indices = []
  other_shape_idx = 0
  if block_info is None:
    start_index_offsets = [None] * len(indices)
  else:
    start_index_offsets = block_info.start_indices
  assert len(indices) + num_mapped_dims == len(full_shape)
  assert len(is_scalar) + num_mapped_dims == len(full_shape)
  assert len(start_index_offsets) == len(full_shape)
  indexer_iter = iter(indices)
  scalar_iter = iter(is_scalar)
  for i, (stride, block_size, dim_size, sio) in enumerate(zip(strides,
                                                              block_shape,
                                                              full_shape,
                                                              start_index_offsets)):
    if block_size is pallas_core.mapped:
      index = tl.core._to_tensor(0, builder)
      is_sc = True
    else:
      index = next(indexer_iter)
      is_sc = next(scalar_iter)
    if isinstance(index, primitives.Slice):
      index_size = index.size
      if isinstance(index.start, int):
        ptr_offset = tl.arange(index.start, index.start + index.size,
                               _builder=builder)
      else:
        ptr_offset = index.start.__add__(tl.arange(0, index.size, _builder=builder),
                                         _builder=builder)
      num_left_expand_dims = len(idx.int_indexer_shape) + other_shape_idx
      num_right_expand_dims = len(other_shape) - other_shape_idx - 1
      other_shape_idx += 1
    elif isinstance(index, slice):
      if index != slice(None):
        raise NotImplementedError("Only `slice(None)` allowed.")
      ptr_offset = tl.arange(0, dim_size, _builder=builder)
      num_left_expand_dims = len(idx.int_indexer_shape) + other_shape_idx
      num_right_expand_dims = len(other_shape) - other_shape_idx - 1
      other_shape_idx += 1
    else:
      # indexer is either a *scalar* or an array of size `int_indexer_shape`
      ptr_offset = index
      num_left_expand_dims = 0
      num_right_expand_dims = len(other_shape)
      if is_sc:
        num_left_expand_dims = max(len(idx.get_indexer_shape()) - 1, 0)
      else:
        num_right_expand_dims = len(other_shape)
    for _ in range(num_left_expand_dims):
      ptr_offset = tl.semantic.expand_dims(ptr_offset, 0, builder)
    for _ in range(num_right_expand_dims):
      ndim = len(ptr_offset.shape)
      ptr_offset = tl.semantic.expand_dims(ptr_offset, ndim, builder)
    # if ptr_offset.shape != desired_shape:
    #   if is_sc:
    #     # Need special handling for reshaping a scalar
    #     ptr_offset = tl.core.broadcast_to(ptr_offset, desired_shape, _builder=builder)
    #   else:
    #     ptr_offset = tl.reshape(ptr_offset, desired_shape, _builder=builder)
    stride_size = tl.core._to_tensor(int(stride), builder)
    if sio:
      ptr_offset = ptr_offset.__add__(sio, _builder=builder)
    bcast_indices.append(ptr_offset.__mul__(stride_size, _builder=builder))
  dest_shape = idx.get_indexer_shape()
  block_shapes = [() if not index.type.is_block() else
                  index.type.get_block_shapes()
                  for index in bcast_indices]
  bcast_indices = [
      tl.core.broadcast_to(index, dest_shape, _builder=builder) if dest_shape != block_shape
      else index for index, block_shape in zip(bcast_indices, block_shapes)]
  for bcast_idx in bcast_indices:
    ptr = ptr.__add__(bcast_idx, _builder=builder)
  return ptr

def _pack_indices(non_slice_idx, indexed_dims):
  non_slice_idx_iter = iter(non_slice_idx)
  return tuple(next(non_slice_idx_iter) if indexed else slice(None) for indexed
               in indexed_dims)

def _get_lowering_rule(ctx: TritonLoweringRuleContext, ptr, *non_slice_idx, indexed_dims):
  ref_block_info, *_ = ctx.block_infos
  idx = _pack_indices(non_slice_idx, indexed_dims)
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  idx_avals = _pack_indices(avals_in[1:], indexed_dims)
  if not isinstance(ptr.type, tl.pointer_type):
    assert len(avals_in) == 1
    return ptr
  if non_slice_idx:
    int_indexer_shape, = {i.shape for i in idx_avals if not isinstance(i, slice)}
  else:
    int_indexer_shape = ()
  is_scalar = [i.shape == () if not isinstance(i, slice) else False for i in
               idx_avals]
  idx = tuple(primitives.Slice.from_slice(slc, s) if isinstance(slc, slice)
              else slc for s, slc in zip(avals_in[0].shape, idx))
  idx = primitives.NDIndexer(idx, avals_in[0].shape, int_indexer_shape)
  ptr = _offset_ptr(ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder, is_scalar)
  return tl.load(ptr, _builder=ctx.builder)
triton_lowering_rules[sp.get_p] = _get_lowering_rule

def _masked_load_lowering_rule(ctx: TritonLoweringRuleContext, ptr,
                               *args, args_tree, masked,
                               eviction_policy, cache_modifier, is_volatile):
  ref_block_info, *_ = ctx.block_infos
  idx, *mask_other = tree_util.tree_unflatten(args_tree, args)
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  if not isinstance(ptr.type, tl.pointer_type):
    assert len(avals_in) == 1
    return ptr
  idx_avals, *_ = tree_util.tree_unflatten(args_tree, avals_in[1:])
  is_scalar = [hasattr(a, "shape") and a.shape == () for a in
               idx_avals.indices]
  ptr = _offset_ptr(ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder, is_scalar)
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
  ref_block_info, *_ = ctx.block_infos
  avals_in = ctx.avals_in
  idx = _pack_indices(non_slice_idx, indexed_dims)
  idx_avals = _pack_indices(avals_in[2:], indexed_dims)
  if non_slice_idx:
    int_indexer_shape, = {i.shape for i in idx_avals if not isinstance(i, slice)}
  else:
    int_indexer_shape = ()
  is_scalar = [i.shape == () if not isinstance(i, slice) else False for i in
               idx]
  idx = tuple(primitives.Slice.from_slice(slc, s) if isinstance(slc, slice)
              else slc for s, slc in zip(avals_in[0].shape, idx))
  idx = primitives.NDIndexer(idx, avals_in[0].shape, int_indexer_shape)
  ptr = _offset_ptr(ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder, is_scalar)
  mask = None
  old_value = tl.load(ptr, mask=mask, _builder=ctx.builder)
  tl.store(ptr, value, mask=mask, _builder=ctx.builder)
  return old_value
triton_lowering_rules[sp.swap_p] = _swap_lowering_rule

def _masked_swap_lowering_rule(ctx: TritonLoweringRuleContext, ptr, value,
                               *args, args_tree, masked, eviction_policy):
  ptr_type = ptr.type.element_ty.element_ty if ptr.type.is_block() else ptr.type.element_ty
  assert ptr_type == value.type.element_ty
  ref_block_info, *_ = ctx.block_infos
  idx, *mask_other = tree_util.tree_unflatten(args_tree, args)
  avals_in = ctx.avals_in
  idx_avals, *_ = tree_util.tree_unflatten(args_tree, avals_in[2:])
  is_scalar = [hasattr(a, "shape") and a.shape == () for a in
               idx_avals.indices]
  ptr = _offset_ptr(ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder, is_scalar)
  mask = None
  if masked:
    assert len(mask_other) == 1
    mask, = mask_other
  return tl.store(ptr, value, mask=mask, _builder=ctx.builder)
triton_lowering_rules[primitives.swap_p] = _masked_swap_lowering_rule

def _addupdate_lowering_rule(ctx: TritonLoweringRuleContext, ptr, value,
    *non_slice_idx, indexed_dims):
  ref_block_info, *_ = ctx.block_infos
  avals_in = ctx.avals_in
  mask = None
  idx = _pack_indices(non_slice_idx, indexed_dims)
  if non_slice_idx:
    int_indexer_shape, = {tuple(map(lambda x: x.value, i.shape)) for i in
                          non_slice_idx}
  else:
    int_indexer_shape = ()
  is_scalar = [i.shape == () if not isinstance(i, slice) else False for i in
               idx]
  idx = tuple(primitives.Slice.from_slice(slc, s) if isinstance(slc, slice)
              else slc for s, slc in zip(avals_in[0].shape, idx))
  idx = primitives.NDIndexer(idx, avals_in[0].shape, int_indexer_shape)
  ptr = _offset_ptr(ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder, is_scalar)
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
  if trans_a:
    a = tl.trans(a, _builder=ctx.builder)
  if trans_b:
    b = tl.trans(b, _builder=ctx.builder)
  allow_tf32 = precision == lax.Precision.HIGH or precision == lax.Precision.DEFAULT
  return tl.dot(a, b, _builder=ctx.builder, allow_tf32=allow_tf32)
triton_lowering_rules[jax.lax.dot_general_p] = _dot_general_lowering

def _reduce_lowering(triton_op, ctx: TritonLoweringRuleContext, a, *, axes):
  if len(axes) != 1:
    raise ValueError("`pallas` reduce operations only support one reduce axis.")
  axis, = axes
  return triton_op(a, axis=axis, _builder=ctx.builder)
triton_lowering_rules[lax.reduce_max_p] = functools.partial(_reduce_lowering, tl.max)
triton_lowering_rules[lax.reduce_min_p] = functools.partial(_reduce_lowering, tl.min)
triton_lowering_rules[lax.reduce_sum_p] = functools.partial(_reduce_lowering, tl.sum)

def _reduce_argmax_lowering(ctx: TritonLoweringRuleContext, a, *, axes,
                            index_dtype):
  if index_dtype != jnp.int32:
    raise ValueError("`index_type` must be f32.")
  if len(axes) != 1:
    raise ValueError("`pallas` reduce operations only support one reduce axis.")
  axis, = axes
  return tl.argmax(a, axis=axis, _builder=ctx.builder)
triton_lowering_rules[lax.argmax_p] = _reduce_argmax_lowering

def _reduce_argmin_lowering(ctx: TritonLoweringRuleContext, a, *, axes,
                            index_dtype):
  if index_dtype != jnp.int32:
    raise ValueError("`index_type` must be f32.")
  if len(axes) != 1:
    raise ValueError("`pallas` reduce operations only support one reduce axis.")
  axis, = axes
  return tl.argmin(a, axis=axis, _builder=ctx.builder)
triton_lowering_rules[lax.argmin_p] = _reduce_argmin_lowering

def _xla_call_lowering_rule(ctx: TritonLoweringRuleContext, *args, call_jaxpr, **_):
  return lower_jaxpr_to_triton_ir(ctx.context, call_jaxpr, *args)
triton_lowering_rules[xla.xla_call_p] = _xla_call_lowering_rule

def _is_read_only(ref_effects) -> bool:
  if len(ref_effects) == 0:
    return True
  if len(ref_effects) > 1:
    # Means we must have a write or accum effect so not read-only
    return False
  eff, = ref_effects
  return isinstance(eff, state.ReadEffect)

def _for_lowering_rule(ctx: TritonLoweringRuleContext, *args, jaxpr,
    which_linear, nsteps, reverse, unroll):
  del which_linear
  if reverse or unroll != 1:
    raise NotImplementedError
  builder = ctx.builder

  lb, ub, s = tl.constexpr(0), tl.constexpr(nsteps), tl.constexpr(1)
  lower_bound = triton.language.core._to_tensor(lb, builder).handle
  upper_bound = triton.language.core._to_tensor(ub, builder).handle
  step = triton.language.core._to_tensor(s, builder).handle

  # Cast ints to MLIR `Index` types
  lower_bound = builder.create_to_index(lower_bound)
  upper_bound = builder.create_to_index(upper_bound)
  step = builder.create_to_index(step)

  current_block = builder.get_insertion_block()

  init_args = args

  # Partially discharge state from jaxpr for non-pointers
  should_discharge = [not isinstance(a, ShapedArrayRef) for a in ctx.avals_in]
  discharged_jaxpr, () = discharge.discharge_state(
      jaxpr, (), should_discharge=[True, *should_discharge])
  in_avals = [v.aval for v in jaxpr.invars[1:]]
  state_effects = state.get_ref_state_effects(in_avals, jaxpr.effects)
  # Read-only `Ref`s don't need to be passed in explicitly as loop arguments so
  # we can filter them out.
  read_only = map(_is_read_only, state_effects)
  is_loop_arg = map(operator.and_, map(operator.not_, read_only), should_discharge)
  ptrs, _ = partition_list(should_discharge, init_args)
  non_loop_args, loop_args = partition_list(is_loop_arg, init_args)

  for_op = builder.create_for_op(lower_bound, upper_bound, step,
                                 [arg.handle for arg in loop_args])
  loop_block = builder.create_block()
  builder.set_insertion_point_to_start(loop_block)

  loop_index = tl.core.tensor(
      builder.create_index_to_si(for_op.get_induction_var()), tl.core.int32)

  # Emit loop body
  for_body_args = [
      tl.core.tensor(for_op.get_body(0).arg(i + 1), arg.type) for i, arg in
      enumerate(loop_args)]
  loop_body_args = merge_lists(is_loop_arg, non_loop_args, for_body_args)
  out_discharged = lower_jaxpr_to_triton_ir(ctx.context, discharged_jaxpr,
                                            [None, *ctx.block_infos],
                                            loop_index, *loop_body_args)
  all_out = merge_lists(should_discharge, ptrs, out_discharged)
  _, loop_out = partition_list(is_loop_arg, all_out)
  if loop_out:
    builder.create_yield_op([arg.handle for arg in loop_out])
  loop_block.merge_block_before(for_op.get_body(0))
  for_results = [for_op.get_result(i) for i in range(len(loop_args))]
  builder.set_insertion_point_to_end(current_block)
  for_out = [tl.core.tensor(r, a.type) for r, a in zip(for_results,
                                                       loop_args)]
  return merge_lists(is_loop_arg, non_loop_args, for_out)
triton_lowering_rules[for_loop.for_p] = _for_lowering_rule

def _while_lowering_rule(ctx: TritonLoweringRuleContext, *args, cond_nconsts,
                         cond_jaxpr, body_nconsts, body_jaxpr):
  raise NotImplementedError
  cond_consts, body_consts, carry = util.split_list(args, [cond_nconsts, body_nconsts])
  cond_const_block_infos, body_const_block_infos, carry_block_infos = util.split_list(
      ctx.block_infos, [cond_nconsts, body_nconsts])
  current_bb = ctx.builder.get_insertion_block()
  loop_bb = _triton.ir.basic_block.create(ctx.builder.context, "loop", current_bb.parent)
  postloop_bb = _triton.ir.basic_block.create(ctx.builder.context, "postloop", current_bb.parent)

  pred, = lower_jaxpr_to_triton_ir(ctx.context, cond_jaxpr.jaxpr,
                                   [*cond_const_block_infos,
                                    *carry_block_infos], *cond_consts, *carry)

  # If pred, we loop otherwise postloop
  ctx.builder.cond_br(pred.handle, loop_bb, postloop_bb)

  # Start populating the loop block
  ctx.builder.set_insert_block(loop_bb)

  instr = loop_bb.get_first_non_phi()
  ctx.builder.set_insert_point((loop_bb, instr))

  old_carry = carry
  lowering_args = []
  for arg in carry:
    lowering_args.append(tl.tensor(ctx.builder.create_phi(arg.type.to_ir(ctx.builder),
                                                          2), arg.type))

  carry = lower_jaxpr_to_triton_ir(ctx.context, body_jaxpr.jaxpr,
                                   [*body_const_block_infos,
                                    *carry_block_infos], *body_consts, *lowering_args)
  pred, = lower_jaxpr_to_triton_ir(ctx.context, cond_jaxpr.jaxpr, *cond_consts, *carry)
  ctx.builder.cond_br(pred.handle, loop_bb, postloop_bb)

  # Update phi nodes to point to outputs of loop block
  bb = lowering_args[0].handle.get_parent()
  prec_bb, next_bb = bb.get_predecessors()
  for old_arg, new_arg, phi_arg in zip(old_carry, carry, lowering_args):
    phi_arg.handle.add_incoming(old_arg.handle, prec_bb)
    phi_arg.handle.add_incoming(new_arg.handle, next_bb)

  # Start populating post loop args
  ctx.builder.set_insert_block(postloop_bb)

  # Set up phi nodes and lget_insertion_blockljkjkpopulate
  post_args = []
  for old_arg, new_arg in zip(old_carry, carry):
    phi_arg = tl.tensor(ctx.builder.create_phi(new_arg.type.to_ir(ctx.builder), 2),
                        new_arg.type)
    phi_arg.handle.add_incoming(old_arg.handle, prec_bb)
    phi_arg.handle.add_incoming(new_arg.handle, next_bb)
    post_args.append(phi_arg)
  return post_args
triton_lowering_rules[lax.while_p] = _while_lowering_rule
