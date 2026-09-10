// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
namespace gjxl::benchmark {
constexpr size_t kGuard = 256;
inline void Check(bool ok, const std::string &what) {
  if (!ok)
    throw std::runtime_error(what);
}
struct Buffer {
  NS::SharedPtr<MTL::Buffer> object;
  size_t size;
  Buffer(MTL::Device *device, size_t bytes) : size(bytes) {
    object = NS::TransferPtr(
        device->newBuffer(bytes + 2 * kGuard, MTL::ResourceStorageModeShared));
    Check(bool(object), "buffer allocation");
    std::memset(object->contents(), 0xa5, bytes + 2 * kGuard);
  }
  void *data() const { return static_cast<char *>(object->contents()) + kGuard; }
  template <class T> T *as() const { return static_cast<T *>(data()); }
  void guards() const {
    const auto *bytes = static_cast<unsigned char *>(object->contents());
    for (size_t i = 0; i < kGuard; ++i)
      Check(bytes[i] == 0xa5 && bytes[kGuard + size + i] == 0xa5,
            "buffer guard modified");
  }
};
struct Kernel {
  NS::SharedPtr<MTL::Library> library;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  Kernel(MTL::Device *device, const char *path,
         const char *name = "gjxl_aq_encode_reconstruction_coefficients") {
    NS::Error *error = nullptr;
    library = NS::TransferPtr(
        device->newLibrary(NS::String::string(path, NS::UTF8StringEncoding), &error));
    Check(bool(library), std::string("load library: ") + path);
    auto function = NS::TransferPtr(
        library->newFunction(NS::String::string(name, NS::UTF8StringEncoding)));
    Check(bool(function), std::string("kernel missing: ") + name);
    pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
    Check(bool(pipeline), std::string("pipeline creation failed: ") + name);
  }
};
} // namespace gjxl::benchmark
