/* // Copyright 2022 The jax_triton Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */

#include "triton_kernel_call.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "cuda.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace jax_triton {

const int TRITON_MAX_N_SHARED_BYTES = 49152;
const int TRITON_MAX_SHARED_OPTIN = 49152;


void TritonExecutable::launch(CUstream stream, void** buffers) {
  CUdevice dev;
  CUcontext ctx;
  // Set the current context to the stream context so we can query the stream
  // device
  cuStreamGetCtx(stream, &ctx);
  cuCtxSetCurrent(ctx);
  /// Only load the kernel if it hasn't already been loaded for this device
  cuCtxGetDevice(&dev);
  CUfunction kernel = load(dev);

  std::vector<void*> params;
  params.reserve(arity);
  for (uint32_t i = 0; i < arity; ++i) {
    params.push_back(&buffers[i]);
  }

  CUresult result =
      cuLaunchKernel(kernel, grid_0, grid_1, grid_2, num_warps * 32, 1, 1,
                     shared_mem, stream, params.data(), /*extra=*/nullptr);
  if (result != 0) {
    std::cout << "Failed launch: " << result << std::endl;
  }
}

CUfunction TritonExecutable::load(CUdevice device) {
  const std::lock_guard<std::mutex> lock(mut);
  if (is_loaded(device)) {
    return kernels[device];
  }
  // Mimics Triton kernel loading
  std::string assembly;
  auto iter = asm_map.find("cubin");
  if (iter != asm_map.end()) {
    assembly = py::cast<std::string>(asm_map["cubin"]);
  } else {
    assert(asm_map.count("ptx") == 1);
    assembly = py::cast<std::string>(asm_map["ptx"]);
  }
  CUfunction fun;
  CUmodule mod;
  cuModuleLoadData(&mod, assembly.c_str());
  cuModuleGetFunction(&fun, mod, name.c_str());

  int shared_optin;
  cuDeviceGetAttribute(&shared_optin,
                       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                       device);
  if (shared_mem > TRITON_MAX_N_SHARED_BYTES &&
      shared_optin > TRITON_MAX_SHARED_OPTIN) {
    cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED);
    int shared_total, shared_static;
    cuDeviceGetAttribute(
        &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device);
    cuFuncGetAttribute(&shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                       fun);
    cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                       shared_optin - shared_static);
  }
  kernels[device] = fun;
  return fun;
}

void do_custom_call(CUstream stream, void** buffers,
    char* opaque, size_t opaque_len) {
  uint64_t descriptor = std::strtoull(opaque, NULL, 0);
  TritonExecutable* executable = TritonExecutable::from_descriptor(descriptor);
  executable->launch(stream, buffers);
}

std::pair<std::string, py::object> MakeTritonExecutable(std::string name, asm_map_t asm_map, uint32_t shared_mem, uint32_t grid_0, uint32_t grid_1, uint32_t grid_2, uint32_t num_warps, uint32_t arity) {
  auto triton_call = std::make_unique<TritonExecutable>(
      name, asm_map, shared_mem, grid_0, grid_1, grid_2, num_warps, arity);
  std::string descriptor = std::to_string(reinterpret_cast<uint64_t>(triton_call.get()));
  py::capsule callback_capsule(triton_call.release(), [](void* ptr) {
    delete reinterpret_cast<TritonExecutable*>(ptr);
  });
  return std::make_pair(descriptor, py::object(std::move(callback_capsule)));
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(reinterpret_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(triton_kernel_call_lib, m) {
  m.def("make_triton_call_descriptor", &MakeTritonExecutable);
  m.def("get_custom_call", [](){
      return EncapsulateFunction(do_custom_call);
      });
}

}  // namespace jax_triton
