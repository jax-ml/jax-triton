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

from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

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

from jax_triton.pallas import tracing_utils

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

Platform = str

@dataclasses.dataclass
class KernelConfig:
  name: Optional[str] = None
  in_specs: Optional[Sequence[Optional[BlockSpec]]] = None
  out_specs: Optional[Sequence[Optional[BlockSpec]]] = None
  grid: Optional[Union[Grid, int]] = None
  meta: dict[str, Any] = dataclasses.field(default_factory=dict)
  compiler_params: dict[Platform, dict[str, Any]] = dataclasses.field(default_factory=dict)

  def replace(self, *args, **kwargs):
    return dataclasses.replace(self, *args, **kwargs)

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
  name: str
  jaxpr: jax_core.Jaxpr
  num_consts: int
  grid_spec: GridSpec
  compiler_params: dict[str, Any]
