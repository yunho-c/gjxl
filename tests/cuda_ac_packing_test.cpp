// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

#include "core/ac_strategy.h"
#include "gpu/cuda/cuda_ac_group_kernels.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::cuda_internal;
constexpr std::array kStrategies = {
  AcStrategyType::kDct8, AcStrategyType::kDct16x16, AcStrategyType::kDct32x32,
  AcStrategyType::kDct16x8, AcStrategyType::kDct8x16,
  AcStrategyType::kDct32x16, AcStrategyType::kDct16x32};
constexpr int32_t kPoison = -7654321;

bool Check(Status status) {
  if (status.ok()) return true;
  std::cerr << status.message() << '\n';
  return false;
}

struct Record {
  size_t x, y, strategy, group, used;
  size_t source = 0, source_stride = 0;
};

bool Case(CudaBackend& backend, Extent2D blocks, size_t pattern) {
  AcStrategyGrid grid;
  if (!Check(AcStrategyGrid::Create(blocks, &grid))) return false;
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      if (grid.occupied(x, y)) continue;
      auto strategy = kStrategies[pattern < 7 ? pattern : (x + 3 * y) % 7];
      const auto covered = GetAcStrategyInfo(strategy)->covered_blocks;
      if (x / 32 != (x + covered.width - 1) / 32 ||
          y / 32 != (y + covered.height - 1) / 32 || !grid.Set(x, y, strategy).ok()) {
        if (!Check(grid.Set(x, y, AcStrategyType::kDct8))) return false;
      }
    }
  }
  const size_t group_width = (blocks.width + 31) / 32;
  const size_t group_count = group_width * ((blocks.height + 31) / 32);
  const size_t count = 3 * blocks.width * blocks.height * 64;
  std::vector<size_t> group_used(group_count, 0), group_size(group_count), group_base(group_count);
  size_t packed = 0;
  for (size_t g = 0; g < group_count; ++g) {
    group_size[g] = std::min<size_t>(32, blocks.width - (g % group_width) * 32) *
      std::min<size_t>(32, blocks.height - (g / group_width) * 32) * 64;
    group_base[g] = packed;
    packed += 3 * group_size[g];
  }
  if (packed != count) return false;
  std::vector<Record> records;
  std::array<std::vector<size_t>, 7> grouped;
  if (!Check(grid.ForEachAnchor([&](size_t x, size_t y, AcStrategyType strategy) {
        const size_t tag = std::find(kStrategies.begin(), kStrategies.end(), strategy) - kStrategies.begin();
        const size_t g = (y / 32) * group_width + x / 32;
        grouped[tag].push_back(records.size());
        records.push_back({x, y, tag, g, group_used[g]});
        group_used[g] += GetAcStrategyInfo(strategy)->coefficient_count();
        return Status::Ok();
      })) || group_used != group_size) return false;
  std::vector<CudaAqAnchor> anchors(7, {9876, 5432});
  std::vector<uint64_t> offsets(7, std::numeric_limits<uint64_t>::max());
  std::array<CudaAqExactBatch, 7> batches{};
  size_t source_offset = 0;
  for (size_t tag = 0; tag < 7; ++tag) {
    const auto* info = GetAcStrategyInfo(kStrategies[tag]);
    auto& batch = batches[tag];
    batch = {static_cast<uint32_t>(anchors.size()), static_cast<uint32_t>(grouped[tag].size()),
      static_cast<uint32_t>(source_offset), static_cast<uint32_t>(info->coefficient_count()),
      static_cast<uint32_t>(info->pixel_extent().width), static_cast<uint32_t>(info->pixel_extent().height),
      static_cast<uint32_t>(info->covered_blocks.width), static_cast<uint32_t>(info->covered_blocks.height)};
    for (size_t index = 0; index < grouped[tag].size(); ++index) {
      Record& record = records[grouped[tag][index]];
      record.source = source_offset + index * batch.coefficient_count;
      record.source_stride = grouped[tag].size() * batch.coefficient_count;
      anchors.push_back({static_cast<uint32_t>(record.x), static_cast<uint32_t>(record.y)});
      offsets.push_back(group_base[record.group] + record.used);
    }
    source_offset += 3 * grouped[tag].size() * batch.coefficient_count;
  }
  if (source_offset != count) return false;
  constexpr size_t source_prefix = 17, output_prefix = 13, final_prefix = 7;
  std::vector<int32_t> source(source_prefix + count + 31, kPoison);
  std::vector<int32_t> expected(output_prefix + count + 29, kPoison), actual(expected.size());
  std::unique_ptr<DeviceBuffer> ds, dd, da, di;
  if (!Check(backend.Allocate(source.size() * 4, &ds)) ||
      !Check(backend.Allocate(expected.size() * 4, &dd)) ||
      !Check(backend.Allocate(anchors.size() * sizeof(CudaAqAnchor), &da)) ||
      !Check(backend.Allocate(offsets.size() * sizeof(uint64_t), &di)) ||
      !Check(backend.CopyHostToDevice(*da, anchors.data(), anchors.size() * sizeof(CudaAqAnchor), 0)) ||
      !Check(backend.CopyHostToDevice(*di, offsets.data(), offsets.size() * sizeof(uint64_t), 0))) return false;
  for (uint32_t reuse = 0; reuse < 3; ++reuse) {
    for (size_t i = 0; i < count; ++i) {
      const uint32_t bits = static_cast<uint32_t>(i) * 1664525u +
        1013904223u + reuse * 987654321u + static_cast<uint32_t>(pattern);
      source[source_prefix + i] = std::bit_cast<int32_t>(bits);
    }
    for (size_t i = 0; i < 5; ++i) {
      constexpr std::array<int32_t, 5> extremes{
        0, -1, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(),
        std::bit_cast<int32_t>(uint32_t{0x81234567})};
      source[source_prefix + i] = extremes[(i + reuse) % extremes.size()];
    }
    for (const Record& record : records) {
      const size_t values = GetAcStrategyInfo(kStrategies[record.strategy])->coefficient_count();
      for (size_t c = 0; c < 3; ++c) {
        std::copy_n(source.data() + source_prefix + record.source + c * record.source_stride, values,
          expected.data() + output_prefix + group_base[record.group] + c * group_size[record.group] + record.used);
      }
    }
    std::fill(actual.begin(), actual.end(), kPoison);
    if (!Check(backend.CopyHostToDevice(*ds, source.data(), source.size() * 4, 0)) ||
        !Check(backend.CopyHostToDevice(*dd, actual.data(), actual.size() * 4, 0))) return false;
    const auto* ap = static_cast<const CudaAqAnchor*>(static_cast<CudaBuffer*>(da.get())->pointer());
    const auto* ip = static_cast<const uint64_t*>(static_cast<CudaBuffer*>(di.get())->pointer());
    const auto* sp = static_cast<const int*>(static_cast<CudaBuffer*>(ds.get())->pointer()) + source_prefix;
    auto* dp = static_cast<int*>(static_cast<CudaBuffer*>(dd.get())->pointer()) + output_prefix;
    for (const auto& batch : batches) {
      if (LaunchCudaPackAcGroups(ap, ip, sp, dp, batch,
            static_cast<uint32_t>(blocks.width), static_cast<uint32_t>(blocks.height), nullptr) != cudaSuccess) return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        !Check(backend.CopyDeviceToHost(*dd, actual.data(), actual.size() * 4, 0)) || actual != expected) return false;
    std::vector<int32_t> final(final_prefix + group_count * 3 * 65536 + 19, kPoison);
    std::vector<CudaDeviceToHostCopy> copies;
    for (size_t g = 0; g < group_count; ++g) {
      copies.push_back({dd.get(), final.data() + final_prefix + g * 3 * 65536,
        group_size[g] * 4, (output_prefix + group_base[g]) * 4, 3, group_size[g] * 4, 65536 * 4});
    }
    if (!Check(backend.CopyDeviceToHostBatch(copies))) return false;
    for (size_t g = 0; g < group_count; ++g) {
      for (size_t c = 0; c < 3; ++c) {
        const int32_t* row = final.data() + final_prefix + (g * 3 + c) * 65536;
        if (!std::equal(row, row + group_size[g], expected.data() + output_prefix + group_base[g] + c * group_size[g]) ||
            !std::all_of(row + group_size[g], row + 65536, [](int32_t v) { return v == kPoison; })) return false;
      }
    }
    if (!std::all_of(final.begin(), final.begin() + final_prefix, [](int32_t v) { return v == kPoison; }) ||
        !std::all_of(final.end() - 19, final.end(), [](int32_t v) { return v == kPoison; })) return false;
    std::vector<int32_t> source_after(source.size());
    if (!Check(backend.CopyDeviceToHost(*ds, source_after.data(), source_after.size() * 4, 0)) || source_after != source) return false;
  }
  std::vector<CudaAqAnchor> anchors_after(anchors.size());
  std::vector<uint64_t> offsets_after(offsets.size());
  return Check(backend.CopyDeviceToHost(*da, anchors_after.data(), anchors_after.size() * sizeof(CudaAqAnchor), 0)) &&
    Check(backend.CopyDeviceToHost(*di, offsets_after.data(), offsets_after.size() * sizeof(uint64_t), 0)) &&
    std::memcmp(anchors.data(), anchors_after.data(), anchors.size() * sizeof(CudaAqAnchor)) == 0 && offsets == offsets_after;
}

bool ReadbackValidation(CudaBackend& backend) {
  std::unique_ptr<DeviceBuffer> buffer;
  std::array<int32_t, 24> input{}, output{};
  for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<int32_t>(i + 1);
  if (!Check(backend.Allocate(sizeof(input), &buffer)) ||
      !Check(backend.CopyHostToDevice(*buffer, input.data(), sizeof(input), 0))) return false;
  const CudaDeviceToHostCopy good{buffer.get(), output.data(), 12, 4, 3, 20, 28};
  std::vector<CudaDeviceToHostCopy> invalid;
  for (size_t field = 0; field < 9; ++field) {
    auto copy = good;
    switch (field) {
      case 0: copy.source = nullptr; break;
      case 1: copy.destination = nullptr; break;
      case 2: copy.source_offset_bytes = sizeof(input) + 1; break;
      case 3: copy.size_bytes = sizeof(input); break;
      case 4: copy.source_row_stride_bytes = 8; break;
      case 5: copy.destination_row_stride_bytes = 8; break;
      case 6: copy.row_count = 9; break;
      case 7: copy.destination_row_stride_bytes = std::numeric_limits<size_t>::max(); break;
      case 8: copy.source_row_stride_bytes = std::numeric_limits<size_t>::max(); break;
    }
    invalid.push_back(copy);
  }
  std::unique_ptr<GpuBackend> other;
  std::unique_ptr<DeviceBuffer> foreign;
  if (!Check(CreateCudaBackend(&other)) || !Check(other->Allocate(sizeof(input), &foreign))) return false;
  auto foreign_copy = good;
  foreign_copy.source = foreign.get();
  invalid.push_back(foreign_copy);
  for (const auto& bad : invalid) {
    output.fill(kPoison);
    const std::array copies{good, bad};
    if (backend.CopyDeviceToHostBatch(copies).code() != StatusCode::kInvalidArgument ||
        !std::all_of(output.begin(), output.end(), [](int32_t v) { return v == kPoison; })) return false;
  }
  output.fill(kPoison);
  if (!Check(backend.CopyDeviceToHostBatch(std::span(&good, 1)))) return false;
  for (size_t i = 0; i < output.size(); ++i) {
    const size_t row = i / 7, column = i % 7;
    const int32_t expected = row < 3 && column < 3 ? input[1 + row * 5 + column] : kPoison;
    if (output[i] != expected) return false;
  }
  auto empty = good;
  empty.size_bytes = 0;
  empty.row_count = std::numeric_limits<size_t>::max();
  empty.destination = nullptr;
  if (!Check(backend.CopyDeviceToHostBatch(std::span(&empty, 1)))) return false;
  empty = good;
  empty.row_count = 0;
  if (!Check(backend.CopyDeviceToHostBatch(std::span(&empty, 1)))) return false;
  CudaAqExactBatch batch{};
  if (LaunchCudaPackAcGroups(nullptr, nullptr, nullptr, nullptr, batch, 0, 0, nullptr) != cudaSuccess) return false;
  batch.anchor_count = 1;
  return LaunchCudaPackAcGroups(nullptr, nullptr, nullptr, nullptr, batch, 0, 0, nullptr) == cudaErrorInvalidValue;
}
}  // namespace

int main(int argc, char** argv) {
  const bool sanitizer = argc == 2 && std::string_view(argv[1]) == "--sanitizer";
  std::unique_ptr<GpuBackend> owner;
  const Status status = CreateCudaBackend(&owner);
  if (status.code() == StatusCode::kUnavailable) return 77;
  if (!Check(status)) return EXIT_FAILURE;
  auto& backend = *static_cast<CudaBackend*>(owner.get());
  if (!ReadbackValidation(backend)) { std::cerr << "Pitched readback validation failed\n"; return EXIT_FAILURE; }
  size_t cases = 0;
  for (Extent2D blocks : {Extent2D{1, 1}, {3, 5}, {31, 29}, {32, 32},
                          {33, 33}, {64, 67}, {240, 135}, {480, 270}}) {
    if (sanitizer && blocks.width > 64) continue;
    for (size_t pattern = 0; pattern < 8; ++pattern) {
      if (!Case(backend, blocks, pattern)) {
        std::cerr << "AC packing failed " << blocks.width << 'x' << blocks.height << " pattern=" << pattern << '\n';
        return EXIT_FAILURE;
      }
      ++cases;
    }
  }
  std::cout << "CUDA AC packing: " << cases << " cases x 3 reuse passes; guarded values, metadata, and pitched copies match.\n" << std::flush;
  return EXIT_SUCCESS;
}
