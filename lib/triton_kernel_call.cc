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

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda.h"
#include "pybind11/pybind11.h"

// TODO(cjfj): Use `Status` for error handling.
#define CHECK_CUDA(expr)                                                  \
  do {                                                                    \
    CUresult result = (expr);                                             \
    if (result != CUDA_SUCCESS) {                                         \
      const char* error_string = "unknown error";                         \
      cuGetErrorString(result, &error_string);                            \
      std::cerr << "CUDA call failed (" << #expr << "): " << error_string \
                << std::endl;                                             \
      abort();                                                            \
    }                                                                     \
  } while (false)

namespace py = pybind11;

namespace jax_triton {
namespace {

constexpr uint32_t kNumThreadsPerWarp = 32;

struct CuModuleDeleter {
  void operator()(CUmodule module) { cuModuleUnload(module); }
};

using OwnedCUmodule =
    std::unique_ptr<std::remove_pointer_t<CUmodule>, CuModuleDeleter>;

class TritonExecutable {
 public:
  TritonExecutable(std::string module_image, std::string kernel_name,
                   uint32_t kernel_arity, uint32_t grid_0, uint32_t grid_1,
                   uint32_t grid_2, uint32_t num_warps,
                   uint32_t shared_mem_bytes)
      : module_image_(std::move(module_image)),
        kernel_name_(std::move(kernel_name)),
        kernel_arity_(kernel_arity),
        grid_{grid_0, grid_1, grid_2},
        shared_mem_bytes_(shared_mem_bytes),
        block_dim_x_(num_warps * kNumThreadsPerWarp) {}

  void Launch(CUstream stream, void** buffers) {
    CUfunction kernel = GetFunctionForCurrentCudaContext();

    std::vector<void*> params;
    params.reserve(kernel_arity_);
    for (uint32_t i = 0; i < kernel_arity_; ++i) {
      params.push_back(&buffers[i]);
    }

    CHECK_CUDA(
        cuLaunchKernel(kernel, grid_[0], grid_[1], grid_[2], block_dim_x_,
                       /*blockDimY=*/1, /*blockDimZ=*/1, shared_mem_bytes_,
                       stream, params.data(), /*extra=*/nullptr));
  }

 private:
  CUfunction GetFunctionForCurrentCudaContext() {
    CUcontext context;
    CHECK_CUDA(cuCtxGetCurrent(&context));

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = functions_.find(context);
    if (it != functions_.end()) {
      return it->second;
    }

    CUmodule module;
    CHECK_CUDA(cuModuleLoadData(&module, module_image_.c_str()));
    modules_.push_back(OwnedCUmodule(module, CuModuleDeleter()));

    CUfunction function;
    CHECK_CUDA(cuModuleGetFunction(&function, module, kernel_name_.c_str()));
    auto [_, success] = functions_.insert({context, function});
    assert(success);

    // The maximum permitted static shared memory allocation in CUDA is 48kB,
    // but we can expose more to the kernel using dynamic shared memory.
    constexpr int kMaxStaticSharedMemBytes = 49152;
    if (shared_mem_bytes_ <= kMaxStaticSharedMemBytes) {
      return function;
    }

    // Set up dynamic shared memory.
    CUdevice device;
    CHECK_CUDA(cuCtxGetDevice(&device));

    int shared_optin;
    CHECK_CUDA(cuDeviceGetAttribute(
        &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));

    if (shared_optin > kMaxStaticSharedMemBytes) {
      CHECK_CUDA(cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED));
      int shared_total;
      CHECK_CUDA(cuDeviceGetAttribute(
          &shared_total,
          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));
      int shared_static;
      CHECK_CUDA(cuFuncGetAttribute(
          &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
      CHECK_CUDA(cuFuncSetAttribute(
          function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          shared_optin - shared_static));
    }
    return function;
  }

  std::string module_image_;
  std::string kernel_name_;
  uint32_t kernel_arity_;
  uint32_t grid_[3];
  uint32_t shared_mem_bytes_;
  uint32_t block_dim_x_;

  std::mutex mutex_;
  std::vector<OwnedCUmodule> modules_;
  std::unordered_map<CUcontext, CUfunction> functions_;
};

}  // namespace

void LaunchTritonExecutable(CUstream stream, void** buffers, char* opaque,
                            size_t opaque_len) {
  assert(opaque_len == sizeof(TritonExecutable*));
  TritonExecutable* executable;
  std::memcpy(&executable, opaque, sizeof(TritonExecutable*));
  executable->Launch(stream, buffers);
}

PYBIND11_MODULE(triton_kernel_call_lib, m) {
  py::class_<TritonExecutable>(m, "TritonExecutable")
      .def(py::init<std::string, std::string, uint32_t, uint32_t, uint32_t,
                    uint32_t, uint32_t, uint32_t>())
      .def_property_readonly("descriptor", [](TritonExecutable& executable) {
        union {
          TritonExecutable* ptr;
          char bytes[sizeof(TritonExecutable*)];
        } descriptor;
        descriptor.ptr = &executable;
        return py::bytes(descriptor.bytes, sizeof(TritonExecutable*));
      });

  m.def("get_custom_call", [] {
    return py::capsule(reinterpret_cast<void*>(&LaunchTritonExecutable),
                       "xla._CUSTOM_CALL_TARGET");
  });
}

}  // namespace jax_triton
