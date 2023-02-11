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

"""Module for calling pallas functions from JAX."""
from functools import partial
import itertools as it
import operator as op

from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import jax
from jax import api_util
from jax import linear_util as lu
from jax import tree_util
from jax import lax
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax._src import ad_util
from jax._src import core as jax_core
from jax._src.lib.mlir.dialects import mhlo
from jax._src import source_info_util
from jax._src import state
from jax._src.state import discharge as state_discharge
from jax._src.util import (
    split_list, safe_map, safe_zip, weakref_lru_cache,
    tuple_insert, partition_list, merge_lists)
from jax._src.lax.control_flow import for_loop
import jax.numpy as jnp
import numpy as np

from jax_triton.pallas import core as pallas_core

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Grid = Tuple[int, ...]
BlockSpec = pallas_core.BlockSpec
BlockMapping = pallas_core.BlockMapping
GridSpec = pallas_core.GridSpec

pallas_call_p = jax_core.Primitive('pallas_call')
pallas_call_p.multiple_results = True

def _maybe_dynamic_slice(start_idx, block_shape, value, is_indexing):
  if start_idx is None:
    assert is_indexing is None
    return value
  assert is_indexing is not None
  output = lax.dynamic_slice(value, start_idx, slice_sizes=block_shape)
  squeeze_dims = tuple(np.arange(len(is_indexing))[np.array(is_indexing,
                                                            dtype=np.bool_)])
  return lax.squeeze(output, squeeze_dims)

def _maybe_dynamic_update_slice(start_idx, block_shape, value, update,
                                is_indexing):
  if start_idx is None:
    assert is_indexing is None
    return update
  assert is_indexing is not None
  broadcast_dims = tuple(i for i, b in enumerate(is_indexing)
                         if not b)
  update = lax.broadcast_in_dim(update, block_shape, broadcast_dims)
  assert update.shape == block_shape
  return lax.dynamic_update_slice(value, update, start_idx)

def _pallas_call_impl(*args, jaxpr, name, out_shapes, which_linear,
                      interpret, debug: bool,
                      in_shapes,
                      input_output_aliases: Tuple[Tuple[int, int], ...],
                      grid_spec: GridSpec,
                      **compiler_params: Any):
  if interpret:
    # If we're in interpreter mode, we *scan* over the grid and eval the
    # discharged jaxpr. This should reproduce exactly what compiling to Triton
    # will do.
    grid = grid_spec.grid
    discharged_jaxpr, consts = state_discharge.discharge_state(jaxpr, ())
    if debug:
      print(discharged_jaxpr)
    loop_indices = jnp.array(list(it.product(*(range(g) for g in grid))))
    oi_map = {v: k for k, v in input_output_aliases}
    out = []
    for i, out_shape in enumerate(out_shapes):
      if i in oi_map:
        out.append(args[oi_map[i]])
      else:
        out.append(jnp.zeros(out_shape.shape, out_shape.dtype))
    carry = [*args, *out]
    def cond(carry):
      return carry[0] < loop_indices.shape[0]
    def body(carry):
      i, *carry = carry
      loop_idx = loop_indices[i]
      start_indices = map(lambda bm: (
        None if bm is None else bm.compute_start_indices(loop_idx)),
        grid_spec.block_mappings)
      block_shapes = [None if block_mapping is None else block_mapping.block_shape
                      for block_mapping in grid_spec.block_mappings]
      is_indexing_dim = [None if bm is None else tuple(b is pallas_core.mapped for b in bm)
                         for bm in block_shapes]
      block_shapes = [None if bm is None else tuple(1 if i else b for i , b in zip(iid, bm))
                      for iid, bm in zip(is_indexing_dim, block_shapes)]
      blocks = map(_maybe_dynamic_slice, start_indices, block_shapes, carry,
                   is_indexing_dim)
      is_mapped_grid_dim = [
          i in grid_spec.mapped_dims for i in range(len(grid_spec.grid))]
      local_grid_env, _ = partition_list(is_mapped_grid_dim,
                                         zip(loop_idx, grid_spec.grid))
      with pallas_core.grid_env(tuple(local_grid_env)):
        blocks = jax.core.eval_jaxpr(discharged_jaxpr, consts, *blocks)
      carry = map(_maybe_dynamic_update_slice, start_indices, block_shapes,
                  carry, blocks, is_indexing_dim)
      return (i + 1, *carry)
    (_, *carry) = lax.while_loop(cond, body, (0, *carry))
    _, out = split_list(carry, [len(args)])
    return out
  return xla.apply_primitive(pallas_call_p, *args, jaxpr=jaxpr, name=name,
                             in_shapes=in_shapes,
                             out_shapes=out_shapes, which_linear=which_linear,
                             grid_spec=grid_spec, interpret=interpret,
                             debug=debug,
                             input_output_aliases=input_output_aliases,
                             **compiler_params)
pallas_call_p.def_impl(_pallas_call_impl)

def _pallas_call_abstract_eval(*avals, out_shapes, **_):
  return map(lambda x: jax_core.ShapedArray(x.shape, x.dtype), out_shapes)
pallas_call_p.def_abstract_eval(_pallas_call_abstract_eval)

def _pallas_call_jvp_rule(primals, tangents, *, jaxpr, name, which_linear,
    input_output_aliases: Tuple[Tuple[int, int], ...],
    in_shapes, out_shapes, grid_spec, debug, interpret, **compiler_params: Any):
  num_inputs = len(in_shapes)
  num_outputs = len(out_shapes)
  if input_output_aliases:
    raise NotImplementedError("JVP with aliasing not supported.")
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  tangents = [ad.instantiate_zeros(t) if inst else t
              for t, inst in zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  nonzero_tangents_with_outputs = nonzero_tangents + [True] * len(out_shapes)
  closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, nonzero_tangents_with_outputs, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  jvp_which_linear = (*which_linear, *(True,) * len(tangents))
  jvp_inshapes = (*in_shapes, *in_shapes)
  jvp_outshapes = (*out_shapes, *out_shapes)
  if input_output_aliases:
    raise NotImplementedError("`input_output_aliases` jvp not supported.")
  # `pallas_call` takes in inputs and returns outputs but its jaxpr *does not*.
  # `pallas_call` takes in a stateful jaxpr, meaning the jaxpr accepts input
  # `Ref`s that are read from followed by output `Ref`s that are written to.
  # This means that when we do `jvp_jaxpr` on the `jaxpr`, we get out a new
  # jaxpr that has tangents following primals. In order for this jaxpr to be
  # compatible w/ `pallas_call` (inputs then outputs), we need to shuffle around
  # the jaxpr's invars.
  logical_primals, logical_tangents = split_list(
      jvp_jaxpr.invars, [len(primals) + len(out_shapes)])
  logical_primal_inputs, logical_primal_outputs = split_list(logical_primals, [len(primals)])
  logical_tangent_inputs, logical_tangent_outputs = split_list(logical_tangents, [len(tangents)])
  in_bms, out_bms = split_list(grid_spec.block_mappings, [len(primals)])
  new_bms = tuple((*in_bms, *in_bms, *out_bms, *out_bms))
  new_grid_spec = grid_spec.replace(block_mappings=new_bms)
  jvp_jaxpr = jvp_jaxpr.replace(invars=[*logical_primal_inputs,
                                        *logical_tangent_inputs,
                                        *logical_primal_outputs,
                                        *logical_tangent_outputs])
  if debug:
    print(jvp_jaxpr)
  out_flat = pallas_call_p.bind(*primals, *tangents, jaxpr=jvp_jaxpr,
                                name=f"{name}_jvp",
                                in_shapes=jvp_inshapes,
                                out_shapes=jvp_outshapes,
                                grid_spec=new_grid_spec,
                                which_linear=jvp_which_linear,
                                interpret=interpret,
                                debug=debug,
                                input_output_aliases=(),
                                **compiler_params)
  out_primals, out_tangents = split_list(out_flat, [len(out_flat) // 2])
  return out_primals, out_tangents
ad.primitive_jvps[pallas_call_p] = _pallas_call_jvp_rule

_save_everything = lambda *_, **__: True

def _convert_outputs_to_writes(
    jaxpr: jax_core.Jaxpr,
    ) -> Tuple[jax_core.Jaxpr, list[jax_core.ShapedArray]]:
  assert not jaxpr.constvars, "Jaxpr shouldn't have constvars."

  in_avals = [v.aval for v in jaxpr.invars]  # [*orig_ref_avals]
  @lu.wrap_init
  def eval_jaxpr(*refs):
    # We split the refs into the original input refs and the dummy residual
    # refs.
    orig_refs, residual_refs = split_list(refs, [len(in_avals)])
    residual_vals = jax_core.eval_jaxpr(jaxpr, (), *orig_refs)
    for res_ref, res_val in zip(residual_refs, residual_vals):
      res_ref[()] = res_val
    return []
  res_ref_avals = [state.ShapedArrayRef(v.aval.shape, v.aval.dtype)  # pytype: disable=attribute-error
                   for v in jaxpr.outvars]
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [*in_avals, *res_ref_avals])
  assert not consts
  return jaxpr, [jax_core.ShapedArray(a.shape, a.dtype) for a in res_ref_avals]

def _convert_inputs_to_reads(num_res: int, jaxpr: jax_core.Jaxpr
                             ) -> jax_core.Jaxpr:
  assert not jaxpr.constvars, "Jaxpr should not have constvars"

  @lu.wrap_init
  def eval_jaxpr(*refs):
    residual_refs, orig_refs = split_list(refs, [num_res])
    residual_vals = [r[()] for r in residual_refs]
    () = jax_core.eval_jaxpr(jaxpr, (), *residual_vals, *orig_refs)
    return []

  res_val_avals, orig_ref_avals = split_list([v.aval for v in jaxpr.invars], [num_res])
  res_ref_avals = [state.ShapedArrayRef(aval.shape, aval.dtype)
                   for aval in res_val_avals]

  jaxpr, _, () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [*res_ref_avals, *orig_ref_avals])
  return jaxpr

def _pallas_call_partial_eval(
    trace: pe.JaxprTrace,
    *tracers: pe.JaxprTracer,
    jaxpr: jax_core.Jaxpr,
    name: str,
    in_shapes: tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
    grid_spec: pallas_core.GridSpec,
    which_linear: tuple[bool, ...],
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    **compiler_params: Any):
  if input_output_aliases:
    raise NotImplementedError
  num_inputs = len(in_shapes)
  num_outputs = len(out_shapes)
  assert num_inputs + num_outputs == len(jaxpr.invars)
  in_unknowns = [not t.pval.is_known() for t in tracers]
  out_unknowns = [False] * num_outputs
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. We want to use the jaxpr to determine which
  # `Ref`s are unknown after executing the for loop body given which `Ref`s are
  # unknown before. However, the jaxpr has no outputs. Instead, we discharge
  # the body and run the fixpoint with the discharged jaxpr. We can do this
  # because the outputs of the jaxpr are one-to-one with the inputs.
  all_in_unknowns = [*in_unknowns, *out_unknowns]
  discharged_jaxpr, discharged_consts = state.discharge_state(jaxpr, ())
  discharged_jaxpr = discharged_jaxpr.replace(
      invars=discharged_jaxpr.constvars + discharged_jaxpr.invars,
      constvars=[])
  for _ in range(num_inputs + num_outputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + all_in_unknowns
    _, _, all_out_unknowns, _, _, = pe.partial_eval_jaxpr_custom(
        discharged_jaxpr, jaxpr_in_unknowns, [True] * len(jaxpr_in_unknowns),
          all_in_unknowns, False, _save_everything)
    all_out_unknowns = list(all_out_unknowns)
    if all_out_unknowns == all_in_unknowns:
      break
    all_in_unknowns = map(op.or_, all_in_unknowns, all_out_unknowns)
  else:
    raise Exception("Invalid fixpoint")
  all_unknowns = all_in_unknowns
  del all_in_unknowns, all_out_unknowns  # redundant since it's the same as `in_unknowns`
  in_unknowns, out_unknowns = split_list(all_unknowns, [num_inputs])

  tracers = tuple(trace.instantiate_const(t) if uk else t  # type: ignore
                  for t, uk in zip(tracers, in_unknowns))

  # We use `partial_eval_jaxpr_custom` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_unknown_resin_, uk_out, inst_out, num_res = \
        pe.partial_eval_jaxpr_custom(
            jaxpr,
            in_inst=all_unknowns,
            in_unknowns=all_unknowns,
            ensure_out_unknowns=[],
            ensure_out_inst=[],
            saveable=_save_everything)
  # # `partial_eval_jaxpr_custom` will give us jaxprs that have hybrid `Ref` and
  # regular valued input/outputs. However, we'd like to bind these jaxprs to a
  # `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.
  # TODO(sharadmv,mattjj): implement "passthrough" optimization.
  # TODO(sharadmv,mattjj): rematerialize loop-dependent values instead of
  # passing the loop index as a residual

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.
  jaxpr_known, res_avals = _convert_outputs_to_writes(jaxpr_known_resout)
  jaxpr_unknown = _convert_inputs_to_reads(num_res, jaxpr_unknown_resin_)

  # Now we execute the forward pass that returns known outputs and residuals
  grid, block_mappings, mapped_dims = (
      grid_spec.grid, grid_spec.block_mappings, grid_spec.mapped_dims)
  in_block_mappings, out_block_mappings = split_list(block_mappings,
                                                     [num_inputs])
  known_in_block_mappings, unknown_in_block_mappings = partition_list(
      in_unknowns, in_block_mappings)
  known_out_block_mappings, unknown_out_block_mappings = partition_list(
      out_unknowns, out_block_mappings)
  known_in_shapes, unknown_in_shapes = partition_list(in_unknowns,
                                                      in_shapes)
  known_out_shapes, unknown_out_shapes = partition_list(out_unknowns,
                                                        out_shapes)
  known_which_linear, unknown_which_linear = partition_list(in_unknowns,
                                                            which_linear)
  res_which_linear = (False,) * num_res
  known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
  known_vals = [t.pval.get_known() for t in known_tracers]
  res_shapes = [jax.ShapeDtypeStruct((*grid, *a.shape), a.dtype)
                for a in res_avals]
  res_index_mappings = [
      jax_core.ClosedJaxpr(
        pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(lambda *args: (*args, *[0] * len(a.shape))),
          [jax_core.ShapedArray((), jnp.int32)] *len(grid))[0], ())
      for a in res_avals
  ]
  res_block_mappings = [
      BlockMapping((*[None] * len(grid), *a.shape), index_map)
      for a, index_map in zip(res_avals, res_index_mappings)
  ]
  known_grid_spec = GridSpec(grid, (*known_in_block_mappings,
                                    *known_out_block_mappings,
                                    *res_block_mappings),
                             grid_spec.mapped_dims)
  unknown_grid_spec = GridSpec(grid, (*res_block_mappings,
                                      *unknown_in_block_mappings,
                                      *unknown_out_block_mappings),
                               grid_spec.mapped_dims)
  known_out_and_res = pallas_call_p.bind(
      *known_vals,
      jaxpr=jaxpr_known,
      grid_spec=known_grid_spec,
      in_shapes=tuple(known_in_shapes),
      out_shapes=(*known_out_shapes, *res_shapes),
      interpret=interpret,
      debug=debug,
      name=f"{name}_known",
      input_output_aliases=(),
      which_linear=tuple(known_which_linear),
      **compiler_params)
  known_outputs, residuals = split_list(known_out_and_res, [len(known_tracers)])
  residuals = map(trace.new_instantiated_const, residuals)
  unknown_inputs = [*residuals, *unknown_tracers]
  unknown_outputs = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(jax_core.ShapedArray(s.shape,
                                                                       s.dtype)), None)
      for s in unknown_out_shapes]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)
  unknown_params = dict(
      jaxpr=jaxpr_unknown,
      in_shapes=(*(jax.ShapeDtypeStruct(s.shape, s.dtype) for s in res_avals),
                 *unknown_in_shapes),
      out_shapes=tuple(unknown_out_shapes),
      grid_spec=unknown_grid_spec,
      which_linear=(*res_which_linear, *unknown_which_linear),
      debug=debug,
      interpret=interpret,
      name=f"{name}_unknown",
      input_output_aliases=(),
      **compiler_params)
  eqn = pe.new_eqn_recipe(unknown_inputs, unknown_outputs,
                          pallas_call_p, unknown_params,
                          jax_core.no_effects, source)
  for t in unknown_outputs: t.recipe = eqn
  return merge_lists(out_unknowns, known_outputs, unknown_outputs)
pe.custom_partial_eval_rules[pallas_call_p] = _pallas_call_partial_eval

def _transpose_jaxpr(jaxpr: jax_core.Jaxpr, which_linear: Sequence[bool]
                     ) -> jax_core.Jaxpr:
  num_inputs = len(which_linear)
  num_outputs = len(jaxpr.invars) - num_inputs
  def trans(*args):
    # First we want to run the computation to read all the residual refs. We can
    # do that by using partial evaluation with all linear inputs unknown.
    res_jaxpr, tangent_jaxpr_, *_ = \
        pe.partial_eval_jaxpr_custom(jaxpr,
                                     in_unknowns=[*which_linear, *[True] *
                                                  num_outputs], 
                                     in_inst=[*which_linear, *[True] *
                                              num_outputs],
                                     ensure_out_inst=[],
                                     ensure_out_unknowns=[],
                                     saveable=_save_everything)
    res_args = [x for x, lin in zip(args, which_linear) if not lin]
    res = jax_core.eval_jaxpr(res_jaxpr, (), *res_args)

    # Now that we have residual values, we run the tangent jaxpr. It takes as
    # input the residuals, and all the refs (at least, the ones
    # that are used in the body). Luckily, `tangent_jaxpr_` has all known and
    # unknown inputs!
    breakpoint()
    primals_args = [*(r for u, r in zip(used_res, res) if u)]
    ct_args = [x for x, u in zip(args, used_ct) if u]
    ad.backward_pass(
        tangent_jaxpr, (), False, (), (*res, *ct_args), ())
    breakpoint()
    return []
  jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(trans), [v.aval for v in jaxpr.invars])
  return jaxpr_trans

def _pallas_call_transpose_rule(cts_in, *args,
                                jaxpr: jax_core.Jaxpr,
                                name: str,
                                in_shapes: Tuple[jax.ShapeDtypeStruct, ...],
                                out_shapes: Tuple[jax.ShapeDtypeStruct, ...],
                                grid_spec: GridSpec,
                                input_output_aliases: Tuple[Tuple[int, int], ...],
                                debug: bool,
                                interpret: bool,
                                which_linear: Tuple[bool, ...],
                                **compiler_params: Any):
  num_inputs = len(in_shapes)
  num_outputs = len(out_shapes)
  is_undefined_primal = [ad.is_undefined_primal(x) for x in args]
  defined_primals, undefined_primals = partition_list(is_undefined_primal, args)
  defined_in_shapes, undefined_in_shapes = partition_list(is_undefined_primal,
                                                          in_shapes)
  block_mappings = grid_spec.block_mappings
  in_block_mappings, out_block_mappings = split_list(block_mappings,
                                                     [num_inputs])
  defined_in_block_mappings, undefined_in_block_mappings = partition_list(
    is_undefined_primal, in_block_mappings)
  defined_which_linear, undefined_which_linear = partition_list(
    is_undefined_primal, which_linear)
  defined_in_shapes, undefined_in_shapes = partition_list(is_undefined_primal,
                                                          in_shapes)
  num_undefined_inputs = sum(is_undefined_primal)
  num_defined_inputs = num_inputs - num_undefined_inputs
  def trans(*args):
    defined_primals, cts, undefined_primals = split_list(args,
                                                         [num_defined_inputs,
                                                          num_outputs])
    # First we want to run the computation to read all the residual refs. We can
    # do that by using partial evaluation with all linear inputs unknown.
    res_jaxpr, tangent_jaxpr_, *_ = \
        pe.partial_eval_jaxpr_custom(jaxpr,
                                     in_unknowns=[*is_undefined_primal, *[True] *
                                                  num_outputs], 
                                     in_inst=[*is_undefined_primal, *[True] *
                                              num_outputs],
                                     ensure_out_inst=[],
                                     ensure_out_unknowns=[],
                                     saveable=_save_everything)
    res = jax_core.eval_jaxpr(res_jaxpr, (), *defined_primals)

    # Now that we have residual values, we run the tangent jaxpr. It takes as
    # input the residuals, and all the refs (at least, the ones
    # that are used in the body). Luckily, `tangent_jaxpr_` has all known and
    # unknown inputs!
    ad.backward_pass(
        tangent_jaxpr_, (), False, (), (*res, *undefined_primals, *cts), ())
    return []
  jaxpr_avals = [v.aval for v in jaxpr.invars]
  jaxpr_in_avals, jaxpr_out_avals = split_list(jaxpr_avals, [num_inputs])
  jaxpr_defined_in_avals, jaxpr_undefined_in_avals = partition_list(
      is_undefined_primal, jaxpr_in_avals)
  jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(trans), [*jaxpr_defined_in_avals, *jaxpr_out_avals,
                            *jaxpr_undefined_in_avals])
  grid_spec = GridSpec(
      grid_spec.grid, (*defined_in_block_mappings, *out_block_mappings,
                       *undefined_in_block_mappings), 
      grid_spec.mapped_dims)
  cts_out = pallas_call_p.bind(
      *defined_primals, *cts_in,
      jaxpr=jaxpr_trans,
      grid_spec=grid_spec,
      in_shapes=(*defined_in_shapes, *out_shapes),
      out_shapes=tuple(undefined_in_shapes),
      name=f"{name}_transpose",
      debug=debug,
      interpret=interpret,
      which_linear=(*defined_which_linear, *[True] * num_outputs),
      input_output_aliases=(),
      **compiler_params)
  cts_out_iter = iter(cts_out)
  return [next(cts_out_iter) if ud else None for 
          ud in is_undefined_primal]
ad.primitive_transposes[pallas_call_p] = _pallas_call_transpose_rule

def _batch_block_mapping(grid: Tuple[int, ...], aval: jax_core.ShapedArray,
                         dim: Union[int, batching.NotMapped],
                         block_mapping: Optional[BlockMapping]) -> BlockMapping:
  def _block_map_function(new_idx, *args):
    if block_mapping is None:
      indices = [0] * len(aval.shape)
    else:
      indices = jax_core.eval_jaxpr(block_mapping.index_map_jaxpr.jaxpr,
                                    block_mapping.index_map_jaxpr.consts,
                                    *args)
    if dim is not batching.not_mapped:
      indices.insert(dim, new_idx)
    return tuple(indices)
  idx_avals = [jax_core.ShapedArray((), jnp.int32)] * (len(grid) + 1)
  block_mapping_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_block_map_function), idx_avals)
  shape = aval.shape if block_mapping is None else block_mapping.block_shape
  if dim is batching.not_mapped:
    new_block_shape = shape
  else:
    new_block_shape = tuple_insert(shape, dim, pallas_core.mapped)
  return BlockMapping(new_block_shape,
                      jax_core.ClosedJaxpr(block_mapping_jaxpr, consts))

def _pallas_call_batching_rule(args, dims, *,
                               jaxpr: jax_core.Jaxpr,
                               name: str,
                               in_shapes: Tuple[jax.ShapeDtypeStruct, ...],
                               out_shapes: Tuple[jax.ShapeDtypeStruct, ...],
                               grid_spec: GridSpec,
                               input_output_aliases: Tuple[Tuple[int, int], ...],
                               debug: bool,
                               interpret: bool,
                               which_linear: Tuple[bool, ...],
                               **compiler_params: Any):
  axis_size, = {x.shape[d] for x, d in zip(args, dims)
                if d is not batching.not_mapped}
  block_mappings = grid_spec.block_mappings
  avals = [v.aval for v in jaxpr.invars]
  # How should we pick output dimensions? This actually matters because XLA
  # can't optimize our pallas kernels, and this layout impacts performance. For
  # now, because `vmap` doesn't really offer a way of inferring good output
  # dimensions. For now, we just use 0.
  # TODO(sharadmv): explore inferring better output dimensions via a heuristic
  # TODO(sharadmv): explore a long term solution to output dim inference

  # When we have input/output aliasing, since the output will be mapped, we need
  # to make sure to broadcast the input across that dimension if it is not
  # mapped.
  dims_ = list(dims)
  args_ = list(args)
  for input_index, _ in input_output_aliases:
    dim = dims_[input_index]
    if dim is batching.not_mapped:
      dims_[input_index] = 0
      args_[input_index] = batching.broadcast(args_[input_index], axis_size, 0)
  args = tuple(args_)
  dims = tuple(dims_)

  all_dims = list(dims) + [0] * len(out_shapes)

  batched_block_mappings = map(partial(_batch_block_mapping, grid_spec.grid),
                               avals, all_dims, block_mappings)
  batched_in_shapes = tuple(
      jax.ShapeDtypeStruct(x.shape if dim is batching.not_mapped else
                           tuple_insert(x.shape, dim, axis_size),
                           x.dtype)
      for x, dim in zip(in_shapes, dims))
  batched_out_shapes = tuple(
      jax.ShapeDtypeStruct(tuple_insert(x.shape, 0, axis_size), x.dtype)
      for x in out_shapes)

  batched_grid_spec = grid_spec.replace(grid=(axis_size, *grid_spec.grid),
                                        block_mappings=tuple(batched_block_mappings),
                                        mapped_dims=(0,) + tuple(a + 1 for a in
                                                                 grid_spec.mapped_dims))
  out = pallas_call_p.bind(*args, jaxpr=jaxpr, name=f"batched_{name}",
                           in_shapes=batched_in_shapes,
                           out_shapes=batched_out_shapes,
                           which_linear=which_linear,
                           grid_spec=batched_grid_spec,
                           input_output_aliases=input_output_aliases,
                           debug=debug,
                           interpret=interpret,
                           **compiler_params)
  return out, (0,) * len(out)
batching.primitive_batchers[pallas_call_p] = _pallas_call_batching_rule

@weakref_lru_cache
def _initial_style_open_jaxpr(fun: Callable, in_tree, in_avals,
                              primitive_name: Optional[str] = None):
  wrapped_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun), in_tree)
  debug = pe.debug_info(fun, in_tree, out_tree_thunk, False,
                        primitive_name or "<unknown>")
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals, debug)
  jaxpr = for_loop._hoist_consts_to_refs(jaxpr)
  return jaxpr, consts, out_tree_thunk()

def _preprocess_grid(grid: Optional[Union[Grid, int]]) -> Grid:
  if grid is None:
    return ()
  if isinstance(grid, int):
    return (grid,)
  return grid

def _extract_function_name(f: Callable, name: Optional[str]) -> str:
  if name is None:
    name = f.__name__ if hasattr(f, "__name__") and f.__name__ else "func"
  return name

def _convert_block_spec_to_block_mapping(
    grid: Grid, block_spec: Optional[BlockSpec]) -> Optional[BlockMapping]:
  if block_spec is None:
    return None
  in_avals = [jax_core.ShapedArray((), jnp.int32) for _ in grid]
  block_shape = tuple(
      pallas_core.mapped if s is None else s for s in block_spec.block_shape)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(block_spec.compute_index), in_avals)
  return BlockMapping(block_shape, jax_core.ClosedJaxpr(jaxpr, consts))

def _compute_shape_from_block_spec(block_spec: Optional[BlockSpec],
                                   arg_shape: Tuple[int, ...]
                                   ) -> Tuple[int, ...]:
  if block_spec is None:
    return arg_shape
  return tuple(s for s in block_spec.block_shape if s is not None)

def pallas_call(f: Callable, out_shape: Any, *, debug: bool = False,
                grid: Optional[Grid] = None,
                in_specs: Optional[Sequence[Optional[BlockSpec]]] = None,
                out_specs: Optional[Sequence[Optional[BlockSpec]]] = None,
                input_output_aliases: Dict[int, int] = {},
                interpret: bool = False,
                name: Optional[str] = None,
                **compiler_params: Any):
  if grid is None:
    if in_specs is not None:
      raise ValueError("Cannot specify `in_specs` with a `None` grid.")
    if out_specs is not None:
      raise ValueError("Cannot specify `out_specs` with a `None` grid.")
  grid = _preprocess_grid(grid)
  name = _extract_function_name(f, name)
  singleton = False
  if not isinstance(out_shape, (tuple, list)):
    out_shape = (out_shape,)
    singleton = True
  if not isinstance(out_shape, tuple):
    out_shape = tuple(out_shape)
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)
  if out_specs is not None and not isinstance(out_specs, (tuple, list)):
    out_specs = (out_specs,)
  if out_specs is not None and not isinstance(out_specs, tuple):
    out_specs = tuple(out_specs)
  flat_out_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype)
                     for x in flat_out_shapes]
  def wrapped(*args):
    flat_args, in_tree = tree_util.tree_flatten(args)
    if grid is None:
      flat_in_specs = [None] * len(flat_args)
      flat_out_specs = [None] * len(flat_out_shapes)
      in_ref_avals = [state.shaped_array_ref(arg.shape, arg.dtype)
                      for arg in flat_args]
      out_ref_avals = [state.shaped_array_ref(arg.shape, arg.dtype)
                       for arg in flat_out_shapes]
    else:
      if in_specs is None:
        flat_in_specs = [None for arg in flat_args]
      else:
        flat_in_specs, in_block_tree = tree_util.tree_flatten(tuple(in_specs))
        if in_block_tree != in_tree:
          raise ValueError(
              "Pytree specs for arguments and `in_specs` must match: "
              f"{in_tree} vs. {in_block_tree}")
      if out_specs is None:
        flat_out_specs = [None for arg in flat_out_shapes]
      else:
        flat_out_specs, out_block_tree = tree_util.tree_flatten(out_specs)
        if out_block_tree != out_tree:
          raise ValueError("Pytree specs for `out_shape` and `out_specs` must match: "
                           f"{out_tree} vs. {out_block_tree}")
      in_ref_avals = [
          state.shaped_array_ref(
            _compute_shape_from_block_spec(block_spec, arg.shape), arg.dtype)
          for block_spec, arg in zip(flat_in_specs, flat_args)]
      out_ref_avals = [
          state.shaped_array_ref(
            _compute_shape_from_block_spec(block_spec, arg.shape), arg.dtype)
          for block_spec, arg in zip(flat_out_specs, flat_out_shapes)]
    in_block_mappings = map(partial(_convert_block_spec_to_block_mapping, grid),
                            flat_in_specs)
    out_block_mappings = map(partial(_convert_block_spec_to_block_mapping, grid),
                             flat_out_specs)
    jaxpr_in_tree = tree_util.tree_structure((*args, *out_shape))
    jaxpr, consts, _ = _initial_style_open_jaxpr(
        f, jaxpr_in_tree, tuple((*in_ref_avals, *out_ref_avals)),
        primitive_name="pallas_call")
    flat_in_specs = it.chain([None] * len(consts), flat_in_specs)
    grid_spec = GridSpec(grid, tuple((*in_block_mappings,
                                      *out_block_mappings)),
                         ())
    which_linear = (False,) * len(flat_args)
    out_flat = pallas_call_p.bind(
        *consts, *flat_args, jaxpr=jaxpr, name=name, which_linear=which_linear,
        in_shapes=tuple(jax.ShapeDtypeStruct(a.shape, a.dtype)
                        for a in flat_args),
        out_shapes=tuple(flat_out_shapes), debug=debug,
        interpret=interpret,
        grid_spec=grid_spec,
        input_output_aliases=tuple(input_output_aliases.items()),
        **compiler_params)
    out = tree_util.tree_unflatten(out_tree, out_flat)
    if singleton:
      return out[0]
    return out
  return wrapped
