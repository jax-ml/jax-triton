# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle

from typing import Any, Callable, Optional, Tuple, Union
from types import SimpleNamespace

import jax
import jax.dlpack
from jax import core
import jax.numpy as jnp
from jax.lib import xla_client as xc
from jax.interpreters import mlir
from jax import tree_util
from jax._src import util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
import numpy as np
import torch
import triton
import triton.language as tl

from jax_triton import custom_call

os.environ["TRITON_CACHE_DIR"] = ""
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

xc.register_custom_call_target("triton_call", custom_call.get_custom_call(), platform="CUDA")

def get_triton_type(obj: Any) -> str:
    type_map = {
        jnp.dtype("float64"): "f64",
        jnp.dtype("float32"): "f32",
        jnp.dtype("float16"): "f16",
        jnp.dtype("int32"): "i32",
        jnp.dtype("int64"): "i64",
    }
    if isinstance(obj, jax.core.ShapedArray):
        return type_map[obj.dtype]
    if isinstance(obj, tl.constexpr):
        obj = obj.value
    if isinstance(obj, int):
        if -2**31 <= obj < 2**31:
            return 'i32'
        elif 2**31 <= obj < 2**32:
            return 'u32'
        elif -2**63 <= obj < 2**63:
            return 'i64'
        elif 2**63 <= obj < 2**64:
            return 'u64'
        else:
            raise ValueError(f'integer overflow representing {obj}')
    if isinstance(obj, float):
        return 'f'
    if isinstance(obj, bool):
        return 'B'
    if isinstance(obj, str):
        return 'str'
    raise NotImplementedError(f'could not compute type name for {obj}')
    
def get_triton_python_ir(aval):
    if aval.shape == ():
        return "scalar", get_triton_type(aval)
    return "ptr", get_triton_type(aval)

def compile(triton_function, constants, *, key, device=0, num_warps=4, num_stages=2):
    def lower(*args):
        arg_types = [get_triton_python_ir(a) for a in args]
        attributes = {i: 16 for i in range(len(args))}
        triton_function._warmup(arg_types=arg_types, device=device,
            attributes=attributes, constants=constants, num_warps=num_warps,
            num_stages=num_stages, key=key, is_manual_warmup=True,
            extern_libs={})
    return lower

def j2t(x_jax):
  x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
  return x_torch

def t2j(x_torch):
  x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
  x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
  return x_jax

Metaparameters = Any
ShapeDtypeDuck = Any

triton_call_p = jax.core.Primitive('triton_call')
triton_call_p.multiple_results = True

def triton_call(*args, kernel, out_shape, grid, num_warps=4, num_stages=2, 
                dump_binary_path: Optional[str] = None, **metaparams):
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)
  out_flat = triton_call_p.bind(*args, kernel=kernel, out_shapes=flat_out_shapes,
      grid=grid, num_warps=num_warps, num_stages=num_stages, 
      dump_binary_path=dump_binary_path, **metaparams)
  return tree_util.tree_unflatten(out_tree, out_flat)

table = {'float32': torch.float32, 'int32': torch.int32, 'float16': torch.float16,
         'float64': torch.float64, 'int64': torch.int64 }

@triton_call_p.def_impl
def triton_call_impl(*args, kernel, out_shapes, grid, dump_binary_path, **metaparams):
  del dump_binary_path
  args_torch = [j2t(x) for x in args]
  outputs_torch = [torch.empty(out_shape.shape, dtype=table[out_shape.dtype.name],
                   device=torch.device('cuda:0')) for out_shape in out_shapes]
  kernel[grid](*args_torch, *outputs_torch, **metaparams)
  return map(t2j, outputs_torch)

@triton_call_p.def_abstract_eval
def triton_call_abstract_eval(*_, out_shapes, **__):
  return [core.ShapedArray(out_shape.shape, out_shape.dtype) for out_shape in out_shapes]

def avals_to_layouts(avals):
  return ir.ArrayAttr.get([aval_to_layout(a) for a in avals])

def aval_to_layout(aval):
  arange = np.arange(aval.ndim, dtype='int64')[::-1].copy()
  return ir.DenseIntElementsAttr.get(arange, type=ir.IndexType.get())

def emit_triton_call(triton_func, avals_in, avals_out, grid, num_warps, num_stages,
                     dump_binary_path: Optional[str], **metaparams):
  metadata = {triton_func.arg_names.index(k) : v for k, v in metaparams.items()}
  compile(triton_func, metadata, num_warps=num_warps, num_stages=num_stages, key="foo")(*avals_in, *avals_out)
  loaded_binary = triton_func.bin_cache["foo"]
  kernel_ptr = loaded_binary.kernel
  shared_mem = loaded_binary.shared_mem
  if dump_binary_path is not None:
    binary = dict(
        asm=loaded_binary.asm,
        shared_mem=shared_mem,
        name=loaded_binary.bin.name)
    with open(dump_binary_path, "wb") as fp:
      pickle.dump(binary, fp)

  grid_ = grid(metaparams)
  grid_0 = grid_[0]
  if len(grid_) == 1:
    grid_1, grid_2 = 1, 1
  elif len(grid_) == 2:
    grid_1, grid_2 = grid_[1], 1
  elif len(grid_) == 3:
    grid_1, grid_2 = grid_[1], grid_[2]
  else:
    assert False
  arity = len(avals_in) + len(avals_out)
  descriptor = custom_call.make_triton_call_descriptor(kernel_ptr, shared_mem, grid_0, grid_1, grid_2, num_warps, arity)
  return descriptor

def triton_call_lowering(ctx, *args, kernel, out_shapes, grid, num_warps=4, num_stages=2,
                         dump_binary_path: Optional[str], **metaparams):
  out_type = ir.TupleType.get_tuple([
      ir.RankedTensorType.get(out_shape.shape, mlir.dtype_to_ir_type(out_shape.dtype))
      for out_shape in out_shapes])
  i32_type = ir.IntegerType.get_signless(32)
  descriptor = emit_triton_call(kernel, ctx.avals_in, ctx.avals_out, grid,
                                num_warps, num_stages, dump_binary_path,
                                **metaparams)
  out = mhlo.CustomCallOp(
            [out_type], args,
            call_target_name=ir.StringAttr.get("triton_call"),
            has_side_effect=ir.BoolAttr.get(False),
            backend_config=ir.StringAttr.get(descriptor),
            api_version=ir.IntegerAttr.get(i32_type, 1),
            called_computations=ir.ArrayAttr.get([]),
            operand_layouts=avals_to_layouts(ctx.avals_in),
            result_layouts=avals_to_layouts(ctx.avals_out))
  results = [mhlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
             for i in range(len(out_shapes))]
  return results
mlir.register_lowering(triton_call_p, triton_call_lowering)
