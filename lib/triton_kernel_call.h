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
#include <mutex>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda.h"

namespace jax_triton {

using asm_map_t = std::unordered_map<std::string, pybind11::object>;

class TritonExecutable {
  public:
   explicit TritonExecutable(std::string name, asm_map_t asm_map, std::uint32_t shared_mem,
                             std::uint32_t grid_0, std::uint32_t grid_1,
                             std::uint32_t grid_2, std::uint32_t num_warps,
                             std::uint32_t arity)
     : name(std::move(name)),
       asm_map(std::move(asm_map)),
       shared_mem(shared_mem),
       grid_0(grid_0),
       grid_1(grid_1),
       grid_2(grid_2),
       num_warps(num_warps),
       arity(arity),
       kernels(),
       mut() {}

   static TritonExecutable* from_descriptor(uint64_t descriptor) {
     return reinterpret_cast<TritonExecutable*>(static_cast<uintptr_t>(descriptor));
   };
   void launch(CUstream stream, void** buffers);

  private:
   bool is_loaded(CUdevice device) const {
     return kernels.count(device) > 0;
   }

   CUfunction load(CUdevice device);
   std::string name;
   asm_map_t asm_map;
   std::uint32_t shared_mem;
   std::uint32_t grid_0;
   std::uint32_t grid_1;
   std::uint32_t grid_2;
   std::uint32_t num_warps;
   std::uint32_t arity;
   std::unordered_map<CUdevice, CUfunction> kernels;
   std::mutex mut;
};

}  // namespace jax_triton
