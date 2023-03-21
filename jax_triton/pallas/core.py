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

"""Module for pallas-core functionality."""
import contextlib
import dataclasses
import functools
from functools import partial

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import state
from jax._src import tree_util
from jax._src.lax.control_flow import for_loop
from jax.interpreters import partial_eval as pe
from jax._src.util import weakref_lru_cache, safe_map, safe_zip
from jax._src.state.types import AbstractRef

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Grid = tuple[int, ...]
GridOrLambda = Union[Callable[..., Grid], Grid]

@dataclasses.dataclass
class GridEnv:
  axis_index: Any
  axis_size: int

_grid_env_stack: List[Tuple[GridEnv, ...]] = []

@contextlib.contextmanager
def grid_env(grid_env: Tuple[Tuple[Any, int], ...]) -> Iterator[None]:
  _grid_env_stack.append(tuple(GridEnv(axis_index, axis_size)
                               for axis_index, axis_size in grid_env))
  try:
    yield
  finally:
    _grid_env_stack.pop()

def current_grid_env() -> Optional[Tuple[GridEnv, ...]]:
  if not _grid_env_stack:
    return None
  return _grid_env_stack[-1]

class Mapped:
  pass
mapped = Mapped()

@dataclasses.dataclass(frozen=True)
class BlockSpec:
  index_map: Callable
  block_shape: Tuple[Optional[int], ...]

  def compute_index(self, *args):
    out = self.index_map(*args)
    if not isinstance(out, tuple):
      out = (out,)
    return out

@dataclasses.dataclass(frozen=True)
class BlockMapping:
  block_shape: Tuple[Union[Mapped, int], ...]
  index_map_jaxpr: jax_core.ClosedJaxpr

  def compute_start_indices(self, loop_idx):
    block_indices = jax_core.jaxpr_as_fun(self.index_map_jaxpr)(*loop_idx)
    return tuple(i if b is mapped else b * i
                 for b, i in zip(self.block_shape, block_indices))

@dataclasses.dataclass(frozen=True)
class GridSpec:
  grid: Tuple[int, ...]
  block_mappings: Tuple[Optional[BlockMapping], ...]
  mapped_dims: Tuple[int, ...]

  replace = dataclasses.replace

@weakref_lru_cache
def _initial_style_open_jaxpr(fun: Callable, in_tree, in_avals,
                              primitive_name: Optional[str] = None):
  wrapped_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun), in_tree)
  debug_info = pe.debug_info(fun, in_tree, out_tree_thunk, False,
                             primitive_name or "<unknown>")
  jaxpr, consts = _initial_style_flat_jaxpr(wrapped_fun, in_avals,
                                            debug_info=debug_info)
  return jaxpr, consts, out_tree_thunk()

def _initial_style_flat_jaxpr(fun: lu.WrappedFun, in_avals,
                              debug_info: Optional[jax_core.DebugInfo] = None
                              ) -> tuple[jax_core.Jaxpr, list[Any]]:
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(fun, in_avals, debug_info)
  jaxpr = for_loop._hoist_consts_to_refs(jaxpr)
  return jaxpr, consts

def preprocess_grid(grid: Optional[Union[Grid, int]]) -> Grid:
  if grid is None:
    return ()
  if isinstance(grid, int):
    return (grid,)
  return grid

def extract_function_name(f: Callable, name: Optional[str]) -> str:
  if name is None:
    name = f.__name__ if hasattr(f, "__name__") and f.__name__ else "func"
  return name

def convert_block_spec_to_block_mapping(
    grid: Grid, block_spec: Optional[BlockSpec]) -> Optional[BlockMapping]:
  if block_spec is None:
    return None
  in_avals = [jax_core.ShapedArray((), jnp.int32) for _ in grid]
  block_shape = tuple(
      mapped if s is None else s for s in block_spec.block_shape)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(block_spec.compute_index), in_avals)
  return BlockMapping(block_shape, jax_core.ClosedJaxpr(jaxpr, consts))

def compute_shape_from_block_spec(block_spec: Optional[BlockSpec],
                                  arg_shape: tuple[int, ...]
                                  ) -> tuple[int, ...]:
  if block_spec is None:
    return arg_shape
  return tuple(s for s in block_spec.block_shape if s is not None)

@dataclasses.dataclass
class SpecializedKernel:
  jaxpr: jax_core.Jaxpr
  grid_spec: GridSpec

@dataclasses.dataclass(frozen=True)
class Kernel:
  func: lu.WrappedFun
  name: Optional[str]
  grid: Optional[Grid]
  in_specs: Optional[list[Optional[BlockSpec]]]
  out_specs: Optional[list[Optional[BlockSpec]]]

  def __post_init__(self):
    if self.grid is None:
      if self.in_specs is not None:
        raise ValueError("Cannot specify `in_specs` with a `None` grid.")
      if self.out_specs is not None:
        raise ValueError("Cannot specify `out_specs` with a `None` grid.")

  def get_name(self) -> str:
    return extract_function_name(self.func, self.name)

  def specialize(self,
                 in_avals: tuple[AbstractRef, ...],
                 out_avals: tuple[AbstractRef, ...],
                 in_tree: tree_util.PyTreeDef
                 ) -> tuple[SpecializedKernel, ...]:
    grid = preprocess_grid(self.grid)
    in_specs = self.in_specs
    out_specs = self.out_specs
    if out_specs is not None and not isinstance(out_specs, (tuple, list)):
      out_specs = (out_specs,)
    if out_specs is not None and not isinstance(out_specs, tuple):
      out_specs = tuple(out_specs)
    if in_specs is None:
      in_specs = [None] * len(in_avals)
    if out_specs is None:
      out_specs = [None] * len(out_avals)
    if grid == ():
      in_ref_avals = [state.shaped_array_ref(arg.shape, arg.dtype)
                      for arg in in_avals]
      out_ref_avals = [state.shaped_array_ref(arg.shape, arg.dtype)
                       for arg in out_avals]
    else:
      in_ref_avals = [
          state.shaped_array_ref(
            compute_shape_from_block_spec(block_spec, aval.shape),
            aval.dtype)
          for block_spec, aval in zip(in_specs, in_avals)]
      out_ref_avals = [
          state.shaped_array_ref(
            compute_shape_from_block_spec(block_spec, aval.shape),
            aval.dtype)
          for block_spec, aval in zip(out_specs, out_avals)]
    in_block_mappings = map(partial(convert_block_spec_to_block_mapping, grid),
                            in_specs)
    out_block_mappings = map(partial(convert_block_spec_to_block_mapping, grid),
                             out_specs)
    grid_spec = GridSpec(grid, (*in_block_mappings, *out_block_mappings), ())
    jaxpr, consts, out_tree = _initial_style_open_jaxpr(
        self.func, in_tree, tuple((*in_ref_avals, *out_ref_avals)))
    return [SpecializedKernel(jaxpr, grid_spec)], consts, out_tree
