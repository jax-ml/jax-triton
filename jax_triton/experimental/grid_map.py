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

"""Module for calling pallas functions from JAX."""
import functools

from typing import Any, Callable, Dict, Sequence, Tuple, Union

import jax
from jax import api_util
from jax import core as jax_core
from jax import lax
from jax import linear_util as lu
from jax import tree_util
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax._src import state
from jax._src.state import primitives as state_primitives
from jax._src.util import safe_map, safe_zip
from jax._src.lax.control_flow import for_loop
import jax.numpy as jnp

from jax_triton.pallas import primitives
from jax_triton.pallas import pallas_call

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

grid_map_p = jax_core.Primitive('grid_map')
grid_map_p.multiple_results = True

grid_map_p.def_impl(functools.partial(xla.apply_primitive, grid_map_p))

def _grid_map_abstract_eval(*avals, out_shapes, **_):
  del avals
  return map(lambda x: jax_core.ShapedArray(x.shape, x.dtype), out_shapes)
grid_map_p.def_abstract_eval(_grid_map_abstract_eval)

def _offset_index(start, idx, block_size):
  if isinstance(idx, slice):
    if idx == slice(None):
      if block_size != 1:
        return start + jnp.arange(block_size)
      return start
    raise NotImplementedError(idx)
  return start + idx

def _compute_block_idx(block_info, idx, out_shape, indexed_dims):
  starts, block_shape = block_info
  idx_iter = iter(idx)
  packed_idx = [next(idx_iter) if indexed else slice(None)
                for indexed in indexed_dims]
  idx = [_offset_index(s, i, b) for s, i, b in zip(starts, packed_idx,
                                                   block_shape)]
  non_slice_idx, indexed_dims = state_primitives._unpack_idx(idx, len(idx))
  idx_iter = iter(range(len(out_shape)))
  if any(i.shape != out_shape for i in non_slice_idx):
    non_slice_idx = [
        lax.broadcast_in_dim(
          idx, out_shape, (next(idx_iter),) if idx.shape != () else ())
        for idx in non_slice_idx]
  return non_slice_idx, indexed_dims

def _run_tiled_jaxpr(jaxpr, block_infos, *refs):
  env = {}
  block_info_env = {}
  def read_env(atom):
    return atom.val if isinstance(atom, jax_core.Literal) else env[atom]
  def write_env(var, val):
    env[var] = val
  map(write_env, jaxpr.invars, refs)
  for invar, info in zip(jaxpr.invars, block_infos):
    if info is not None:
      block_info_env[invar] = info
  for eqn in jaxpr.eqns:
    if eqn.primitive == state.get_p:
      ref, *non_slice_idx = map(read_env, eqn.invars)
      params = eqn.params
      if eqn.invars[0] in block_info_env:
        block_info = block_info_env[eqn.invars[0]]
        out_shape = eqn.outvars[0].aval.shape
        non_slice_idx, indexed_dims = _compute_block_idx(
            block_info, non_slice_idx, out_shape, eqn.params["indexed_dims"])
        params = dict(eqn.params, indexed_dims=indexed_dims)
      ans = eqn.primitive.bind(ref, *non_slice_idx, **params)
      write_env(eqn.outvars[0], ans)
    elif eqn.primitive == state.swap_p:
      ref, x, *non_slice_idx = map(read_env, eqn.invars)
      params = eqn.params
      if eqn.invars[0] in block_info_env:
        block_info = block_info_env[eqn.invars[0]]
        out_shape = eqn.outvars[0].aval.shape
        non_slice_idx, indexed_dims = _compute_block_idx(
            block_info, non_slice_idx, out_shape, eqn.params["indexed_dims"])
        params = dict(eqn.params, indexed_dims=indexed_dims)
      ans = eqn.primitive.bind(ref, x, *non_slice_idx, **params)
      write_env(eqn.outvars[0], ans)
    elif eqn.primitive == for_loop.for_p:
      refs = map(read_env, eqn.invars)
      local_block_infos = [block_info_env.get(v, None) for v in eqn.invars]
      def new_body(i, *args):
        for info, arg in zip(local_block_infos, args):
          if info is not None:
            assert len(info[0]) == len(arg.shape)
        return _run_tiled_jaxpr(eqn.params["jaxpr"], [None, *local_block_infos],
                                 i, *args)
      new_avals = [state.ShapedArrayRef(x.shape, x.dtype) for x in refs]
      new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(new_body), [jax_core.ShapedArray((), jnp.int32),
                                   *new_avals])
      new_jaxpr = for_loop._hoist_consts_to_refs(new_jaxpr)
      new_params = dict(eqn.params, jaxpr=new_jaxpr)
      ans = for_loop.for_p.bind(*consts, *refs, **new_params)
      map(write_env, eqn.outvars, ans[len(consts):])
    else:
      ans = eqn.primitive.bind(*map(read_env, eqn.invars), **eqn.params)
      if eqn.primitive.multiple_results:
        map(write_env, eqn.outvars, ans)
      else:
        write_env(eqn.outvars[0], ans)
  return map(read_env, jaxpr.outvars)

def _maybe_tuple(s: Union[Any, Tuple[Any, ...]]) -> Tuple[Any, ...]:
  if not isinstance(s, tuple):
    return (s,)
  return s

def grid_map(f,
             out_shape: Any,
             grid: Union[int, Tuple[int, ...]],
             input_block_shapes: Sequence[Tuple[int, ...]],
             output_block_shapes: Sequence[Tuple[int, ...]],
             input_index_map: Dict[int, Callable[[int, ...], Union[int, Tuple[int, ...]]]],
             output_index_map: Dict[int, Callable[[int, ...], Union[int, Tuple[int, ...]]]],
             *, debug: bool = False, **params: Any):
  if isinstance(grid, int):
    grid = (grid,)
  flat_out_shapes, _ = tree_util.tree_flatten(out_shape)

  def wrapped(*refs):
    flat_output_avals = [jax_core.ShapedArray(out_shape.shape, out_shape.dtype)
        for out_shape in flat_out_shapes]
    ptr_avals = [state.ShapedArrayRef(block_shape, ref.dtype)
                 for block_shape, ref in zip(input_block_shapes, refs)]
    out_ptr_avals = [state.ShapedArrayRef(block_shape, aval.dtype) for aval,
                     block_shape in zip(flat_output_avals, output_block_shapes)]
    in_tree = tree_util.tree_structure([*ptr_avals, *out_ptr_avals])
    flat_fun, _ = api_util.flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, [*ptr_avals, *out_ptr_avals])
    jaxpr = for_loop._hoist_consts_to_refs(jaxpr)

    @functools.wraps(f)
    def run_kernel(*refs):
      idx = [primitives.program_id(axis=i) for i, _ in enumerate(grid)]
      input_start_indices = [_maybe_tuple(start_index(*idx))
                             for start_index in input_index_map]
      output_start_indices = [_maybe_tuple(start_index(*idx))
                              for start_index in output_index_map]
      block_infos = [(start_idx, block_shape)
                     for start_idx, block_shape in zip([*input_start_indices,
                                                        *output_start_indices],
                                                       [*input_block_shapes,
                                                         *output_block_shapes])]
      block_infos = [None] * len(consts) + block_infos
      # The jaxpr for `f` is written for *tiles* not for the entire array. For
      # Triton, we need to convert the tiled indexing into full-array indexing.
      _run_tiled_jaxpr(jaxpr, block_infos, *refs)

    return pallas_call(
        run_kernel,
        out_shape=[jax.ShapeDtypeStruct(s.shape, s.dtype) for s in flat_out_shapes],
        debug=debug, grid=grid, **params)(*consts, *refs)
  return wrapped
