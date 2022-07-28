/* Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>
#include <string>

#include <pybind11/pybind11.h>
#include "cuda.h"

namespace py = pybind11;

template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
  return std::string(reinterpret_cast<const char*>(&descriptor), sizeof(T));
}

template <typename T>
void UnpackDescriptor(T* descriptor_ptr, const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::invalid_argument( "received negative value" );
  }
  std::memcpy(descriptor_ptr, opaque, opaque_len);
}

struct TritonCallDescriptor {
  CUfunction kernel_ptr;
  std::uint32_t shared_mem;
  std::uint32_t grid_0;
  std::uint32_t grid_1;
  std::uint32_t grid_2;
  std::uint32_t num_warps;
  std::uint32_t arity;
};

void do_custom_call(CUstream stream, void** buffers,
    char* opaque, size_t opaque_len) {
  TritonCallDescriptor descriptor;
  UnpackDescriptor(&descriptor, opaque, opaque_len);
  CUfunction kernel = descriptor.kernel_ptr;
  int grid_0 = descriptor.grid_0;
  int grid_1 = descriptor.grid_1;
  int grid_2 = descriptor.grid_2;
  int num_warps = descriptor.num_warps;
  int arity = descriptor.arity;
  std::string params;
  params.resize(8 * arity);
  char* params_ptr = &params[0];
  for (int i = 0; i < arity; i++) {
    params_ptr = (char*)(((uintptr_t)params_ptr + 7) & (-8));
    std::memcpy(params_ptr, &buffers[i], 8);
    params_ptr += 8;
  }
  size_t params_size = (std::ptrdiff_t)(params_ptr - &params[0]);
  void* config[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, params.data(),
    CU_LAUNCH_PARAM_BUFFER_SIZE, &params_size,
    CU_LAUNCH_PARAM_END
  };
  CUresult result = cuLaunchKernel(kernel, grid_0, grid_1, grid_2, num_warps * 32, 1, 1, descriptor.shared_mem, stream, nullptr, config);
  if (result != 0) {
    std::cout << "Failed launch: " << result << std::endl;
  }
  // cuStreamSynchronize(stream);
}

std::string MakeTritonCallDescriptor(uint64_t kernel_ptr, uint32_t shared_mem, uint32_t grid_0, uint32_t grid_1, uint32_t grid_2, uint32_t num_warps, uint32_t arity) {
  TritonCallDescriptor descriptor;
  descriptor.kernel_ptr = reinterpret_cast<CUfunction>(kernel_ptr);
  descriptor.shared_mem = shared_mem;
  descriptor.grid_0 = grid_0;
  descriptor.grid_1 = grid_1;
  descriptor.grid_2 = grid_2;
  descriptor.num_warps = num_warps;
  descriptor.arity = arity;
  return PackDescriptorAsString(descriptor);
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(reinterpret_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(custom_call, m) {
  m.def("make_triton_call_descriptor", [](uint64_t kernel_ptr, uint32_t shared_mem, uint32_t grid_0, uint32_t grid_1, uint32_t grid_2, uint32_t num_warps, uint32_t arity){ return py::bytes(MakeTritonCallDescriptor(kernel_ptr, shared_mem, grid_0, grid_1, grid_2, num_warps, arity));
      });
  m.def("get_custom_call", [](){ return EncapsulateFunction(do_custom_call); });
}
