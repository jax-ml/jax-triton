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
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "cuda.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

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

class TritonKernel {
 public:
  TritonKernel(std::string module_image, std::string kernel_name,
               uint32_t num_warps, uint32_t shared_mem_bytes)
      : module_image_(std::move(module_image)),
        kernel_name_(std::move(kernel_name)),
        block_dim_x_(num_warps * kNumThreadsPerWarp),
        shared_mem_bytes_(shared_mem_bytes) {}

  void Launch(CUstream stream, uint32_t grid[3], void** params) {
    CUfunction kernel = GetFunctionForCurrentCudaContext();
    CHECK_CUDA(cuLaunchKernel(kernel, grid[0], grid[1], grid[2], block_dim_x_,
                              /*blockDimY=*/1, /*blockDimZ=*/1,
                              shared_mem_bytes_, stream, params,
                              /*extra=*/nullptr));
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
  uint32_t block_dim_x_;
  uint32_t shared_mem_bytes_;

  std::mutex mutex_;
  std::vector<OwnedCUmodule> modules_;
  std::unordered_map<CUcontext, CUfunction> functions_;
};

class TritonKernelCall {
 public:
  TritonKernelCall(std::shared_ptr<TritonKernel> kernel, uint32_t grid_0,
                   uint32_t grid_1, uint32_t grid_2,
                   std::vector<std::optional<uint64_t>> parameters)
      : kernel_(std::move(kernel)),
        grid_{grid_0, grid_1, grid_2},
        parameters_(std::move(parameters)) {}

  void Launch(CUstream stream, void** buffers) {
    std::vector<void*> params;
    params.reserve(parameters_.size());
    for (std::optional<uint64_t>& param : parameters_) {
      if (param.has_value()) {
        params.push_back(&*param);
      } else {
        params.push_back(buffers++);
      }
    }

    kernel_->Launch(stream, grid_, params.data());
  }

 private:
  std::shared_ptr<TritonKernel> kernel_;
  uint32_t grid_[3];
  // Parameter values. `nullopt` values represent buffer arguments.
  std::vector<std::optional<uint64_t>> parameters_;
};

template <typename CppT, typename PyT>
uint64_t EncodeKernelParameterAs(PyT value) {
  static_assert(sizeof(CppT) <= sizeof(uint64_t));
  union {
    CppT value;
    uint64_t bits;
  } encoded;
  encoded.bits = 0;
  encoded.value = CppT(value);
  return encoded.bits;
}

uint64_t EncodeKernelParameter(py::int_ value, std::string_view dtype) {
  if ((dtype == "i1") || (dtype == "i8")) {
    return EncodeKernelParameterAs<int8_t>(value);
  } else if (dtype == "u8") {
    return EncodeKernelParameterAs<uint8_t>(value);
  } else if (dtype == "i16") {
    return EncodeKernelParameterAs<int16_t>(value);
  } else if (dtype == "u16") {
    return EncodeKernelParameterAs<uint16_t>(value);
  } else if (dtype == "i32") {
    return EncodeKernelParameterAs<int32_t>(value);
  } else if (dtype == "u32") {
    return EncodeKernelParameterAs<uint32_t>(value);
  } else if (dtype == "i64") {
    return EncodeKernelParameterAs<int64_t>(value);
  } else if (dtype == "u64") {
    return EncodeKernelParameterAs<uint64_t>(value);
  } else {
    throw std::runtime_error(std::string("unknown dtype: ") + dtype.data());
  }
}

uint64_t EncodeKernelParameter(py::float_ value, std::string_view dtype) {
  if (dtype == "fp32") {
    return EncodeKernelParameterAs<float>(value);
  } else if (dtype == "fp64") {
    return EncodeKernelParameterAs<double>(value);
  } else {
    throw std::runtime_error(std::string("unknown dtype: ") + dtype.data());
  }
}

uint64_t EncodeKernelParameter(py::bool_ value, std::string_view dtype) {
  if ((dtype == "int1") || (dtype == "B")) {
    return EncodeKernelParameterAs<bool>(value);
  } else {
    throw std::runtime_error(std::string("unknown dtype: ") + dtype.data());
  }
}

}  // namespace

void LaunchTritonKernel(CUstream stream, void** buffers, char* opaque,
                        size_t opaque_len) {
  assert(opaque_len == sizeof(TritonKernelCall*));
  TritonKernelCall* kernel_call;
  std::memcpy(&kernel_call, opaque, sizeof(TritonKernelCall*));
  kernel_call->Launch(stream, buffers);
}

PYBIND11_MODULE(triton_kernel_call_lib, m) {
  py::class_<TritonKernel, std::shared_ptr<TritonKernel>>(m, "TritonKernel")
      .def(py::init<std::string, std::string, uint32_t, uint32_t>());

  py::class_<TritonKernelCall>(m, "TritonKernelCall")
      .def(py::init<std::shared_ptr<TritonKernel>, uint32_t, uint32_t, uint32_t,
                    std::vector<std::optional<uint64_t>>>())
      .def_property_readonly("descriptor", [](TritonKernelCall& kernel_call) {
        union {
          TritonKernelCall* ptr;
          char bytes[sizeof(TritonKernelCall*)];
        } descriptor;
        descriptor.ptr = &kernel_call;
        return py::bytes(descriptor.bytes, sizeof(TritonKernelCall*));
      });

  m.def("get_custom_call", [] {
    return py::capsule(reinterpret_cast<void*>(&LaunchTritonKernel),
                       "xla._CUSTOM_CALL_TARGET");
  });

  m.def("encode_kernel_parameter",
        py::overload_cast<py::int_, std::string_view>(&EncodeKernelParameter));
  m.def(
      "encode_kernel_parameter",
      py::overload_cast<py::float_, std::string_view>(&EncodeKernelParameter));
  m.def("encode_kernel_parameter",
        py::overload_cast<py::bool_, std::string_view>(&EncodeKernelParameter));
}

}  // namespace jax_triton
