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

"""Module for lowering JAX primitives to Triton IR."""
import dataclasses
import functools

from typing import Any, Dict, Optional, NamedTuple, Tuple, Sequence

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
from jax.interpreters import mlir
from jax._src import core as jax_core
from jax._src import pjit
from jax._src import state
from jax.lib import xla_client as xc
from jax._src.state import primitives as sp
from jax._src.state import discharge
from jax._src.state import AbstractRef
from jax._src.util import weakref_lru_cache
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
import jax.numpy as jnp
import numpy as np
import triton
import triton.language as tl
from triton.language import ir as tl_ir
import triton.libtriton.triton as _triton

from jax_triton import triton_lib
from jax_triton import triton_kernel_call_lib
from jax_triton import utils as triton_utils
from jax_triton.pallas import primitives
from jax_triton.pallas import core as pallas_core
from jax_triton.pallas import pallas_call_p
from jax_triton.triton_lib import get_triton_type

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip
partial = functools.partial

Grid = Tuple[int, ...]
BlockSpec = pallas_core.BlockSpec
BlockMapping = pallas_core.BlockMapping
GridSpec = pallas_core.GridSpec


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
  grid: Tuple[int, ...]

@dataclasses.dataclass
class TritonCompilationResult:
  name: str
  asm: Dict[str, str]
  shared_mem: int
  lowering_result: TritonLoweringResult

def _eval_index_map(ctx: TritonModuleContext, idx, block_mapping: Optional[BlockMapping]):
  if block_mapping is None:
    return None
  block_indices = tuple(lower_jaxpr_to_triton_ir(
    ctx, block_mapping.index_map_jaxpr.jaxpr, None, *idx))
  return tuple(
      i if b is pallas_core.mapped else i.__mul__(b, _builder=ctx.builder)
      for i, b in zip(block_indices, block_mapping.block_shape))

triton_lowering_rules = {}

def _process_grid_to_3d_grid(builder, grid_spec: GridSpec):
  if len(grid_spec.grid) <= 3:
    program_ids = [tl.program_id(axis=i, _builder=builder)
                   for i in range(len(grid_spec.grid))]
    return grid_spec.grid, program_ids
  grid_prefix = grid_spec.grid[:-2]
  grid_suffix = grid_spec.grid[-2:]
  total_axis_size = np.prod(grid_prefix)
  new_grid = (total_axis_size, *grid_suffix)
  out_indices = [0] * len(grid_prefix)
  grid0 = tl.program_id(0, _builder=builder)
  for i, s in reversed(list(enumerate(grid_prefix))):
    grid0, out_indices[i] = (
        grid0.__floordiv__(s, _builder=builder),
        grid0.__mod__(s, _builder=builder))
  out_indices = [*out_indices, tl.program_id(1, _builder=builder),
                 tl.program_id(2, _builder=builder)]
  assert len(out_indices) == len(grid_spec.grid)
  return new_grid, out_indices

def lower_jaxpr_to_triton_module(jaxpr: jax_core.Jaxpr, num_consts: int, in_shapes, grid_spec: GridSpec,
                                 name: str) -> tl_ir.module:
  jaxpr, _ = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars), instantiate=True)
  ir_context = tl_ir.context()
  builder = tl_ir.builder(ir_context)
  module = tl_ir.module("", builder)

  in_avals = [var.aval for var in jaxpr.invars]
  triton_types = [get_triton_type(x) for x in in_avals]
  arg_types = [triton.compiler.str_to_ty(arg) for arg in triton_types]
  assert len(jaxpr.outvars) == 0
  ret_type = tl.void
  prototype = tl.function_type(ret_type, arg_types)
  out = prototype.to_ir(builder)
  fn = module.get_or_insert_function(name, out)
  args = []
  for i in range(len(in_avals)):
    fn.add_attr(i + 1, tl_ir.attribute(tl_ir.attribute_kind.aligned, 16))
    ptr = tl.tensor(fn.args[i], prototype.param_types[i])
    args.append(ptr)
  fn.set_is_kernel(True)
  insert_pt = builder.get_insert_block()
  entry = tl_ir.basic_block.create(builder.context, "entry", fn)
  builder.set_insert_block(entry)
  new_grid, program_ids = _process_grid_to_3d_grid(builder, grid_spec)
  local_program_ids = [pid for i, pid in enumerate(program_ids)
                       if i not in grid_spec.mapped_dims]
  ctx = TritonModuleContext(name, ir_context, builder, module, grid_spec,
                            local_program_ids)
  start_indices = map(partial(_eval_index_map, ctx, program_ids),
                      grid_spec.block_mappings)
  arg_block_infos = [BlockInfo(shape,
                     start_idx, block_mapping.block_shape)
                     if block_mapping is not None else None
                     for shape, block_mapping, start_idx in
                     zip(in_shapes, grid_spec.block_mappings, start_indices)]
  const_block_infos = [None] * num_consts
  block_infos = [*const_block_infos, *arg_block_infos]
  () = lower_jaxpr_to_triton_ir(ctx, jaxpr, block_infos, *args)
  ctx.builder.ret_void()
  ctx.builder.set_insert_block(insert_pt)
  return TritonLoweringResult(ir_context, module, builder, new_grid)

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

def _atomic_cas_lowering_rule(ctx: TritonLoweringRuleContext, ptr, cmp, val):
  return tl.atomic_cas(ptr, cmp, val, _builder=ctx.builder)
triton_lowering_rules[primitives.atomic_cas_p] = _atomic_cas_lowering_rule

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

def eq_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__eq__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.eq_p] = eq_lowering_rule

def bitwise_and_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__and__(b, _builder=ctx.builder)
triton_lowering_rules[jax.lax.and_p] = bitwise_and_lowering_rule

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
  # Add dummy dimensions
  if not a.type.is_block():
    return tl.broadcast_to(a, list(shape), _builder=ctx.builder)
  a_shape_iter = iter(a.shape)
  new_shape = [next(a_shape_iter) if i in broadcast_dimensions else 1
               for i in range(len(shape))]
  new_shape = tuple(tl.constexpr(v) for v in new_shape)
  a = tl.reshape(a, new_shape, _builder=ctx.builder)
  return tl.broadcast_to(a, list(shape), _builder=ctx.builder)
triton_lowering_rules[jax.lax.broadcast_in_dim_p] = _broadcast_in_dim_lowering_rule

def _squeeze_lowering_rule(ctx: TritonLoweringRuleContext, a, *, dimensions):
  del dimensions
  shape = [tl.constexpr(s) for s in ctx.avals_out[0].shape]
  return tl.reshape(a, shape, _builder=ctx.builder)
triton_lowering_rules[jax.lax.squeeze_p] = _squeeze_lowering_rule

def _reshape_lowering_rule(ctx: TritonLoweringRuleContext, a, *, new_sizes,
                           dimensions):
  del new_sizes, dimensions
  shape = [tl.constexpr(s) for s in ctx.avals_out[0].shape]
  return tl.reshape(a, shape, _builder=ctx.builder)
triton_lowering_rules[jax.lax.reshape_p] = _reshape_lowering_rule

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
  strides = triton_utils.strides_from_shape(full_shape)
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
        index = tl.arange(index.start, index.start + index.size,
                          _builder=builder)
      else:
        index = index.start.__add__(tl.arange(0, index.size, _builder=builder),
                                    _builder=builder)
      desired_shape = ([tl.constexpr(1)] * other_shape_idx + [tl.constexpr(index_size)] +
                       [tl.constexpr(1)] * (len(other_shape) - other_shape_idx - 1))
      desired_shape = [tl.constexpr(1)] * len(idx.int_indexer_shape) + desired_shape
      other_shape_idx += 1
    elif isinstance(index, slice):
      if index != slice(None):
        raise NotImplementedError("Only `slice(None)` allowed.")
      index = tl.arange(0, dim_size, _builder=builder)
      desired_shape = ([tl.constexpr(1)] * other_shape_idx + [tl.constexpr(dim_size)] +
                       [tl.constexpr(1)] * (len(other_shape) - other_shape_idx - 1))
      desired_shape = [tl.constexpr(1)] * len(idx.int_indexer_shape) + desired_shape
      other_shape_idx += 1
    else:
      if is_sc:
        desired_shape = [tl.constexpr(1)] * max(len(idx.get_indexer_shape()), 1)
      else:
        desired_shape = index.shape + [tl.constexpr(1)] * len(other_shape)
    if index.shape != desired_shape:
      if is_sc:
        # Need special handling for reshaping a scalar
        index = tl.broadcast_to(index, desired_shape, _builder=builder)
      else:
        index = tl.reshape(index, desired_shape, _builder=builder)
    stride_size = tl.core._to_tensor(int(stride), builder)
    if sio:
      index = index.__add__(sio, _builder=builder)
    bcast_indices.append(index.__mul__(stride_size, _builder=builder))
  dest_shape = map(tl.constexpr, idx.get_indexer_shape())
  if dest_shape == []:
    # We can't have ()-shaped arrays in Triton.
    dest_shape = [tl.constexpr(1)]
  bcast_indices = [
      tl.broadcast_to(index, dest_shape, _builder=builder) if dest_shape != index.shape
      else index for index in bcast_indices]
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
                               *args, args_tree, masked,
                               eviction_policy):
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
  return tl.store(ptr, value, mask=mask, eviction_policy=eviction_policy,
                  _builder=ctx.builder)
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
  allow_tf32 = precision == lax.Precision.HIGH or precision == lax.Precision.DEFAULT
  return tl.dot(a, b, _builder=ctx.builder, trans_a=trans_a, trans_b=trans_b,
                allow_tf32=allow_tf32)
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

def _pjit_lowering_rule(ctx: TritonLoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_triton_ir(ctx.context, jaxpr.jaxpr, ctx.block_infos, *args)
triton_lowering_rules[pjit.pjit_p] = _pjit_lowering_rule

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
  should_discharge = [not isinstance(a, AbstractRef) for a in ctx.avals_in]
  loop_counter = ctx.builder.create_phi(tl.int32.to_ir(ctx.builder), 2)
  ref_avals = [v.aval for v in jaxpr.invars]
  read_only = [for_loop._is_read_only(eff) if eff else True for eff in
               state.get_ref_state_effects(ref_avals, jaxpr.effects)][1:]
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
                                      [None, *ctx.block_infos],
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

  # Start populating post loop args
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

def _while_lowering_rule(ctx: TritonLoweringRuleContext, *args, cond_nconsts,
                         cond_jaxpr, body_nconsts, body_jaxpr):
  cond_consts, body_consts, carry = util.split_list(args, [cond_nconsts, body_nconsts])
  cond_const_block_infos, body_const_block_infos, carry_block_infos = util.split_list(
      ctx.block_infos, [cond_nconsts, body_nconsts])
  current_bb = ctx.builder.get_insert_block()
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
  pred, = lower_jaxpr_to_triton_ir(ctx.context, cond_jaxpr.jaxpr,
                                   [*cond_const_block_infos,
                                    *carry_block_infos], *cond_consts, *carry)
  ctx.builder.cond_br(pred.handle, loop_bb, postloop_bb)

  # Update phi nodes to point to outputs of loop block
  bb = lowering_args[0].handle.get_parent()
  prec_bb, next_bb = bb.get_predecessors()
  for old_arg, new_arg, phi_arg in zip(old_carry, carry, lowering_args):
    phi_arg.handle.add_incoming(old_arg.handle, prec_bb)
    phi_arg.handle.add_incoming(new_arg.handle, next_bb)

  # Start populating post loop args
  ctx.builder.set_insert_block(postloop_bb)

  # Set up phi nodes and populate
  post_args = []
  for old_arg, new_arg in zip(old_carry, carry):
    phi_arg = tl.tensor(ctx.builder.create_phi(new_arg.type.to_ir(ctx.builder), 2),
                        new_arg.type)
    phi_arg.handle.add_incoming(old_arg.handle, prec_bb)
    phi_arg.handle.add_incoming(new_arg.handle, next_bb)
    post_args.append(phi_arg)
  return post_args
triton_lowering_rules[lax.while_p] = _while_lowering_rule

@weakref_lru_cache
def compile_jaxpr(jaxpr: jax_core.Jaxpr, num_consts: int, in_shapes, grid_spec: GridSpec,
                   name: str, num_warps: int, num_stages: int
                   ) -> TritonCompilationResult:
  lowering_result = lower_jaxpr_to_triton_module(jaxpr, num_consts, in_shapes, grid_spec, name)
  backend = _triton.runtime.backend.CUDA
  device = 0
  name, asm, shared_mem = _triton.code_gen.compile_ttir(
      backend, lowering_result.module, device, num_warps, num_stages, {}, 0)
  return TritonCompilationResult(name, asm, shared_mem, lowering_result)


def pallas_call_lowering(ctx: mlir.LoweringRuleContext, *in_nodes,
                         kernels: Sequence[pallas_core.SpecializedKernel],
                         name: str,
                         in_shapes: Tuple[jax.ShapeDtypeStruct, ...],
                         out_shapes: Tuple[jax.ShapeDtypeStruct, ...],
                         which_linear: Tuple[bool, ...],
                         interpret: bool,
                         debug: bool,
                         input_output_aliases: Tuple[Tuple[int, int], ...]):
  if interpret:
    return mlir.lower_fun(pallas_call_p.impl, multiple_results=True)(
        ctx, *in_nodes, kernels=kernels, name=name, out_shapes=out_shapes,
        in_shapes=in_shapes,
        which_linear=which_linear,
        interpret=interpret, debug=debug,
        input_output_aliases=input_output_aliases)
  lowered_kernels = []
  for kernel in kernels:
    if debug:
      print(kernel.jaxpr)
      print(kernel.grid_spec)
    compiler_params = kernel.compiler_params
    num_warps = compiler_params.get("num_warps", 4)
    num_stages = compiler_params.get("num_stages", 3)
    compilation_result = compile_jaxpr(kernel.jaxpr, kernel.num_consts,
                                       tuple((*in_shapes, *out_shapes)),
                                       kernel.grid_spec, kernel.name, num_warps, num_stages)
    name = compilation_result.name
    asm = compilation_result.asm
    shared_mem = compilation_result.shared_mem
    lowering_result = compilation_result.lowering_result
    if debug:
      lowering_result.module.print()
    lowered_kernels.append((name, asm, shared_mem, lowering_result))
  out_type = ir.TupleType.get_tuple([
      ir.RankedTensorType.get(out_shape.shape, mlir.dtype_to_ir_type(out_shape.dtype))
      for out_shape in ctx.avals_out])
  i32_type = ir.IntegerType.get_signless(32)

  if len(lowered_kernels) == 1:
    name, asm, shared_mem, lowering_result = lowered_kernels[0]
    kernel = triton_kernel_call_lib.TritonKernel(
        asm["cubin"], name, num_warps, shared_mem
    )
    grid = triton_utils.normalize_grid(
        lowering_result.grid, metaparams={})
    # All arguments are buffers.
    kernel_params = []
    for _ in range(len(in_shapes) + len(out_shapes)):
      kernel_params.append(
          triton_kernel_call_lib.create_array_parameter(
              0,  # bytes to zero  # TODO(cjfj): Expose through user API.
              True,  # divisible by 16
          )
      )
    kernel_call = triton_kernel_call_lib.TritonKernelCall(
        kernel, grid[0], grid[1], grid[2], kernel_params
    )
  elif len(lowered_kernels) > 1:
    kernel_calls = []
    for name, asm, shared_mem, lowering_result in lowered_kernels:
      kernel = triton_kernel_call_lib.TritonKernel(
          asm["cubin"], name, num_warps, shared_mem
      )
      grid = triton_utils.normalize_grid(
          lowering_result.grid, metaparams={})
      # All arguments are buffers.
      kernel_params = []
      for _ in range(len(in_shapes) + len(out_shapes)):
        kernel_params.append(
            triton_kernel_call_lib.create_array_parameter(
                0,  # bytes to zero  # TODO(cjfj): Expose through user API.
                True,  # divisible by 16
            )
        )
      kernel_call = triton_kernel_call_lib.TritonKernelCall(
          kernel, grid[0], grid[1], grid[2], kernel_params
      )
      kernel_calls.append(kernel_call)
    input_output_aliases_with_sizes = tuple(
        (input_idx, output_idx, triton_lib.aval_size_bytes(ctx.avals_in[input_idx]))
        for input_idx, output_idx in input_output_aliases
    )

    kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
          name,
          [(call, f"config{i}") for i, call in enumerate(kernel_calls)],
          input_output_aliases_with_sizes,
    )
  else:
    raise ValueError("Cannot have 0 kernels.")

  ctx.module_context.add_keepalive(kernel_call)
  output_operand_aliases = ir.ArrayAttr.get([
          mhlo.OutputOperandAlias.get(
              output_tuple_indices=[output],
              operand_index=input,
              operand_tuple_indices=[])
          for input, output in input_output_aliases
      ])
  out = mhlo.CustomCallOp(
      [out_type],
      in_nodes,
      call_target_name=ir.StringAttr.get("triton_kernel_call"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(kernel_call.descriptor),
      api_version=ir.IntegerAttr.get(i32_type, 1),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=triton_utils.avals_to_layouts(ctx.avals_in),
      result_layouts=triton_utils.avals_to_layouts(ctx.avals_out),
      output_operand_aliases=output_operand_aliases,
  )
  results = [mhlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
             for i in range(len(out_shapes))]
  return results
mlir.register_lowering(pallas_call_p, pallas_call_lowering, platform="cuda")

xc.register_custom_call_target(
  "triton_kernel_call", triton_kernel_call_lib.get_custom_call(), platform="CUDA")
