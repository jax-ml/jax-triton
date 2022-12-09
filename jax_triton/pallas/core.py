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

"""Module for pallas-core functionality."""
import contextlib
import dataclasses

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

from jax import core as jax_core

@dataclasses.dataclass
class GridEnv:
  axis_index: Any
  axis_size: int

_grid_env_stack: List[Tuple[GridEnv, ...]] = []

@contextlib.contextmanager
def grid_env(grid_env: Tuple[Tuple[Any, int], ...]) -> Iterator[None]:
  _grid_env_stack.append(tuple(GridEnv(axis_index, axis_size)
                               for axis_index, axis_size in grid_env))
  yield
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
    jaxpr = self.index_map_jaxpr.jaxpr
    consts = self.index_map_jaxpr.consts
    block_indices = jax_core.eval_jaxpr(jaxpr, consts, *loop_idx)
    return tuple(i if b is mapped else b * i
                 for b, i in zip(self.block_shape, block_indices))

@dataclasses.dataclass(frozen=True)
class GridSpec:
  grid: Tuple[int, ...]
  block_mappings: Tuple[Optional[BlockMapping], ...]
  mapped_dims: Tuple[int, ...]

  replace = dataclasses.replace
