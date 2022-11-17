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
from functools import partial
import itertools as it

from typing import Callable, Dict, NamedTuple, Optional, Tuple

import jax
from jax import api_util
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax import lax
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax.interpreters import xla
from jax._src import ad_util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax._src import state
from jax._src.util import (
    split_list, safe_map, safe_zip, weakref_lru_cache)
from jax._src.lax.control_flow import for_loop
import jax.numpy as jnp

from triton._C.libtriton import triton as tc

from jax_triton.triton_call import emit_triton_kernel_call, avals_to_layouts
from jax_triton.pallas import lowering
from jax_triton.pallas import core as pallas_core

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

pallas_call_p = jax_core.Primitive('pallas_call')
pallas_call_p.multiple_results = True

pallas_call_p.def_impl(partial(xla.apply_primitive, pallas_call_p))

def _pallas_call_impl(*args, jaxpr, name, out_shapes, which_linear, grid,
                      num_warps, num_stages, interpret, debug: bool,
                      input_output_aliases: Tuple[Tuple[int, int], ...],
                      **metaparams):
  if interpret:
    # If we're in interpreter mode, we *scan* over the grid and eval the
    # discharged jaxpr. This should reproduce exactly what compiling to Triton
    # will do.
    discharged_jaxpr, consts = state.discharge_state(jaxpr, ())
    if debug:
      print(discharged_jaxpr)
    loop_indices = jnp.array(list(it.product(*(range(g) for g in grid))))
    grid_sizes = grid
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
      with pallas_core.grid_env(tuple(zip(loop_idx, grid_sizes))):
        carry = jax.core.eval_jaxpr(discharged_jaxpr, consts, *carry)
      return (i + 1, *carry)
    (_, *carry) = lax.while_loop(cond, body, (0, *carry))
    _, out = split_list(carry, [len(args)])
    return out
  return xla.apply_primitive(pallas_call_p, *args, jaxpr=jaxpr, name=name,
                             out_shapes=out_shapes, which_linear=which_linear,
                             grid=grid, num_warps=num_warps,
                             num_stages=num_stages, interpret=interpret,
                             debug=debug,
                             input_output_aliases=input_output_aliases,
                             **metaparams)
pallas_call_p.def_impl(_pallas_call_impl)

def _pallas_call_abstract_eval(*avals, out_shapes, **_):
  return map(lambda x: jax_core.ShapedArray(x.shape, x.dtype), out_shapes)
pallas_call_p.def_abstract_eval(_pallas_call_abstract_eval)

def _pallas_call_jvp_rule(primals, tangents, *, jaxpr, name, which_linear,
    input_output_aliases: Tuple[Tuple[int, int], ...],
    out_shapes, grid, debug, interpret, **metaparams):
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
  jvp_jaxpr = jvp_jaxpr.replace(invars=[*logical_primal_inputs,
                                        *logical_tangent_inputs,
                                        *logical_primal_outputs,
                                        *logical_tangent_outputs])
  if debug:
    print(jvp_jaxpr)
  out_flat = pallas_call_p.bind(*primals, *tangents, jaxpr=jvp_jaxpr,
                                name=f"{name}_jvp",
                                out_shapes=jvp_outshapes,
                                which_linear=jvp_which_linear,
                                interpret=interpret,
                                grid=grid, debug=debug,
                                input_output_aliases=(), **metaparams)
  out_primals, out_tangents = split_list(out_flat, [len(out_flat) // 2])
  return out_primals, out_tangents
ad.primitive_jvps[pallas_call_p] = _pallas_call_jvp_rule

class TritonCompilationResult(NamedTuple):
  name: str
  asm: Dict[str, str]
  shared_mem: int
  lowering_result: lowering.TritonLoweringResult

@weakref_lru_cache
def _compile_jaxpr(jaxpr: jax_core.Jaxpr, name: str, num_warps: int,
                   num_stages: int) -> TritonCompilationResult:
  lowering_result = lowering.lower_jaxpr_to_triton_module(jaxpr, name)
  backend = tc.runtime.backend.CUDA
  device = 0
  name, asm, shared_mem = tc.code_gen.compile_ttir(backend, lowering_result.module, device,
      num_warps, num_stages, {}, 0)
  return TritonCompilationResult(name, asm, shared_mem, lowering_result)


def pallas_call_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, jaxpr, name,
                         out_shapes, which_linear, grid, num_warps, num_stages,
                         interpret,
                         debug: bool,
                         input_output_aliases: Tuple[Tuple[int, int], ...],
                         **metaparams):
  if interpret:
    return mlir.lower_fun(_pallas_call_impl, multiple_results=True)(
        ctx, *in_nodes, jaxpr=jaxpr, name=name, out_shapes=out_shapes,
        which_linear=which_linear, grid=grid, num_warps=num_warps,
        num_stages=num_stages, interpret=interpret, debug=debug,
        input_output_aliases=input_output_aliases,
        **metaparams)
  compilation_result = _compile_jaxpr(jaxpr, name, num_warps, num_stages)
  name = compilation_result.name
  asm = compilation_result.asm
  shared_mem = compilation_result.shared_mem
  lowering_result = compilation_result.lowering_result
  if debug:
    lowering_result.module.print()
  out_type = ir.TupleType.get_tuple([
      ir.RankedTensorType.get(out_shape.shape, mlir.dtype_to_ir_type(out_shape.dtype))
      for out_shape in ctx.avals_out])
  i32_type = ir.IntegerType.get_signless(32)
  descriptor, keepalive = emit_triton_kernel_call(
      ctx, name, asm, shared_mem, num_warps=num_warps, grid=grid,
      metaparams=metaparams, dump_binary_path=None)
  ctx.module_context.add_keepalive(keepalive)
  output_operand_aliases = ir.ArrayAttr.get([
          mhlo.OutputOperandAlias.get(
              output_tuple_indices=[output],
              operand_index=input,
              operand_tuple_indices=[])
          for input, output in input_output_aliases
      ])
  out = mhlo.CustomCallOp(
            [out_type], in_nodes,
            call_target_name=ir.StringAttr.get("triton_kernel_call"),
            has_side_effect=ir.BoolAttr.get(False),
            backend_config=ir.StringAttr.get(descriptor),
            api_version=ir.IntegerAttr.get(i32_type, 1),
            called_computations=ir.ArrayAttr.get([]),
            operand_layouts=avals_to_layouts(ctx.avals_in),
            result_layouts=avals_to_layouts(ctx.avals_out),
            output_operand_aliases=output_operand_aliases)
  results = [mhlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
             for i in range(len(out_shapes))]
  return results
mlir.register_lowering(pallas_call_p, pallas_call_lowering)

@weakref_lru_cache
def _initial_style_open_jaxpr(fun: Callable, in_tree, in_avals,
                              primitive_name: Optional[str] = None):
  wrapped_fun, out_tree = api_util.flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  debug = pe.debug_info(fun, in_tree, False, primitive_name or "<unknown>")
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals, debug)
  jaxpr = for_loop._hoist_consts_to_refs(jaxpr)
  return jaxpr, consts, out_tree()

def clear_caches():
  _initial_style_open_jaxpr.cache_clear()
  _compile_jaxpr.cache_clear()

def pallas_call(f, out_shape, grid, *, debug: bool = False,
                num_warps: int = 4, num_stages: int = 3,
                input_output_aliases: Dict[int, int] = {},
                interpret: bool = False,
                **metaparams):
  if isinstance(grid, int):
    grid = (grid,)
  name = f.__name__ if hasattr(f, "__name__") and f.__name__ else"func"
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)
  flat_out_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in
                     flat_out_shapes]
  def wrapped(*args, **kwargs):
    flat_args, _ = tree_util.tree_flatten((args, kwargs))
    in_tree = tree_util.tree_structure((*args, *flat_out_shapes))
    flat_avals = map(
        jax_core.raise_to_shaped, map(jax_core.get_aval, flat_args))
    flat_output_avals = [jax_core.ShapedArray(out_shape.shape, out_shape.dtype)
        for out_shape in flat_out_shapes]
    ptr_avals = [state.ShapedArrayRef(aval.shape, aval.dtype) for aval in flat_avals]
    out_ptr_avals = [state.ShapedArrayRef(aval.shape, aval.dtype) for aval in flat_output_avals]
    jaxpr, consts, _ = _initial_style_open_jaxpr(
        f, in_tree, tuple((*ptr_avals, *out_ptr_avals)), primitive_name="pallas_call")
    if debug:
      print(jaxpr)

    which_linear = (False,) * len(flat_args)
    out_flat = pallas_call_p.bind(
        *consts, *flat_args, jaxpr=jaxpr, name=name, which_linear=which_linear,
        out_shapes=tuple(flat_out_shapes), grid=grid, debug=debug,
        interpret=interpret,
        num_warps=num_warps,
        input_output_aliases=tuple(input_output_aliases.items()),
        num_stages=num_stages, **metaparams)
    return tree_util.tree_unflatten(out_tree, out_flat)
  return wrapped
