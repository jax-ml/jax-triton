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
"""Module for tracing utilities."""

from typing import Any, Callable, Optional

from jax.interpreters import partial_eval as pe
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src.util import weakref_lru_cache, safe_map, safe_zip, HashablePartial
from jax._src.lax.control_flow import for_loop

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

@weakref_lru_cache
def initial_style_open_jaxpr(fun: Callable, in_tree, in_avals, primitive_name, *args, **kwargs):
  wrapped_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(HashablePartial(fun, *args, **kwargs)), in_tree)
  debug_info = pe.debug_info(fun, in_tree, out_tree_thunk, False,
                             primitive_name or "<unknown>")
  jaxpr, consts = initial_style_flat_jaxpr(wrapped_fun, in_avals,
                                           debug_info=debug_info)
  return jaxpr, consts, out_tree_thunk()

def initial_style_flat_jaxpr(fun: lu.WrappedFun, in_avals,
                              debug_info: Optional[jax_core.JaxprDebugInfo] = None
                              ) -> tuple[jax_core.Jaxpr, list[Any]]:
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(fun, in_avals, debug_info)
  jaxpr = for_loop._hoist_consts_to_refs(jaxpr)
  return jaxpr, consts

