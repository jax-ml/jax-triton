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
from jax._src import state
from jax._src.state import discharge as state_discharge
from jax._src.util import (
    split_list, safe_map, safe_zip, weakref_lru_cache,
    tuple_insert, partition_list)
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
  jvp_which_linear = which_linear + (True,) * len(tangents)
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

def pallas_call(f: Callable, *, out_shape: Any, debug: bool = False,
                grid: Optional[Grid] = None,
                in_specs: Optional[Sequence[Optional[BlockSpec]]] = None,
                out_specs: Optional[Sequence[Optional[BlockSpec]]] = None,
                input_output_aliases: Dict[int, int] = {},
                interpret: bool = False,
                name: Optional[str] = None,
                **compiler_params: Any) -> Any:
  """Executes a Pallas kernel on input JAX arrays.

  Args:
    f: A kernel function. It should accept input `Ref`s first, followed by
      output `Ref`s afterwards.
    out_shape: A Pytree of `ShapeDtypeStruct`s (or values that have `.shape` and
      `.dtype` properties). An output `Ref` is passed into `f` for each value
      in `out_shape`.
    debug: Prints out debugging information about the kernel when `True`.
    interpret: Emulates kernel execution in HLO if `True`.
    name: A string name for the kernel.
    grid: An optional tuple of integers that determines the schedule by which
      the kernel is executed. If not provided, the grid is empty.
    in_specs: a list of optional `BlockSpecs` for each of the inputs to the
      kernel.
    out_specs: a list of optional `BlockSpecs` for each of the outputs from the
      kernel.
    input_output_aliases: A mapping of input indices to output indices,
      indicating which inputs should be aliased to which outputs.
    **compiler_params: A dictionary mapping compiler name to compiler options.
  Returns:
    A Pytree of JAX Arrays
  """
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
  @jax.jit
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
