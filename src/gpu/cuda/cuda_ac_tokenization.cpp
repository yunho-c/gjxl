// SPDX-License-Identifier: Apache-2.0
#include "gpu/cuda/cuda_ac_tokenization.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

#include "codestream/representation_storage_plan.h"
#include "codestream/simple_ac_context.h"
#include "core/frame_geometry.h"
#include "core/overwrite_array.h"
#include "gpu/cuda/cuda_ac_tokenization_kernels.h"
#include "gpu/cuda/cuda_backend_internal.h"

namespace gjxl::cuda_internal {
using namespace codestream_internal;
namespace {
using Frame = vardct_frame_internal::VarDctFrameView;
struct Failure {
  Status status;
};
void Check(Status status) {
  if (!status.ok()) throw Failure{std::move(status)};
}
void Require(bool ok, const char* message) {
  if (!ok) throw Failure{Status::InvalidArgument(message)};
}
struct Plan {
  Storage<CudaTokenAnchor> anchors;
  Storage<CudaTokenGroup> groups;
  Storage<CudaTokenStrategy> strategies;
  Storage<uint32_t> orders, control;
  uint32_t map_size = 0, maximum_tokens = 0, tasks = 0, coefficient_count = 0;
  size_t end = 0, context_count = 0;
  bool scalar = false;
};
Plan MakePlan(const Frame& frame, const SimpleCoefficientOrders& orders,
              const SimpleAcNaturalOrders& natural,
              const SimpleBlockContextMap& map, bool collect,
              uint32_t capacity_hint) {
  Check(ValidateSimpleBlockContextMap(map));
  Plan p;
  const auto blocks = frame.geometry().block_grid().blocks;
  const uint64_t block_count = uint64_t(blocks.width) * blocks.height;
  Require(block_count && block_count <= UINT32_MAX / 195u,
          "Token index bound exceeded");
  Require(map.num_contexts && map.num_contexts <= 65535 / 495 &&
              map.context_map.size() == 3 * 13 * (map.qf_thresholds.size() + 1),
          "Context shape unsupported");
  p.anchors.reserve(static_cast<size_t>(block_count));
  p.orders.reserve(3 * 2624);
  p.control.reserve(9 + map.qf_thresholds.size() + map.context_map.size());
  p.maximum_tokens = static_cast<uint32_t>(195 * block_count);
  p.context_count = map.ac_context_count();
  p.strategies.resize(kAcStrategyCount);
  for (size_t i = 0; i < kAcStrategyCount; ++i) {
    if (natural.orders[i].empty()) continue;
    Require(
        i == 0 || i == 4 || i == 5 || i == 6 || i == 7 || i == 10 || i == 11,
        "Unsupported CUDA token strategy family");
    const auto& info = *GetAcStrategyInfo(static_cast<AcStrategyType>(i));
    auto& s = p.strategies[i];
    s.count = static_cast<uint32_t>(info.coefficient_count());
    s.width = static_cast<uint32_t>(info.covered_blocks.width);
    s.height = static_cast<uint32_t>(info.covered_blocks.height);
    s.covered = s.width * s.height;
    s.log2covered = std::countr_zero(s.covered);
    const auto family = kSimpleStrategyOrder[i];
    uint32_t offsets[3];
    for (size_t c = 0; c < 3; ++c) {
      const auto& scan = (orders.used_order_mask & (uint16_t{1} << family))
                             ? orders.orders[family][c]
                             : natural.orders[i];
      Require(scan.size() == s.count, "Token order size differs");
      std::array<bool, 1024> seen{};
      Require(s.count <= seen.size(), "Token order bound");
      for (const auto value : scan) {
        Require(value < s.count && !seen[value],
                "Invalid coefficient permutation");
        seen[value] = true;
      }
      offsets[c] = static_cast<uint32_t>(p.orders.size());
      p.orders.insert(p.orders.end(), scan.begin(), scan.end());
    }
    s.order0 = offsets[0];
    s.order1 = offsets[1];
    s.order2 = offsets[2];
  }
  Require(p.orders.size() <= 3 * 2624, "Token scan capacity exceeded");
  p.groups.resize(frame.ac_group_count());
  uint32_t source = 0;
  bool mixed = false;
  for (uint32_t g = 0; g < p.groups.size(); ++g) {
    VarDctNativeAcGroupView native;
    Check(frame.GetNativeAcGroup(g, &native));
    std::visit(
        [&](const auto& view) {
          auto& group = p.groups[g];
          group.first = static_cast<uint32_t>(p.anchors.size());
          group.width = static_cast<uint32_t>(view.block_extent.width);
          group.height = static_cast<uint32_t>(view.block_extent.height);
          group.map = p.map_size;
          p.map_size += 3 * group.width * group.height;
          const uint32_t stride =
              static_cast<uint32_t>(view.used_coefficient_count);
          uint32_t anchor_source = source;
          for (uint32_t y = 0; y < group.height; ++y)
            for (uint32_t x = 0; x < group.width; ++x) {
              AcStrategyCell cell;
              Check(frame.strategies().Get(view.block_x + x, view.block_y + y,
                                           &cell));
              if (!cell.is_anchor) continue;
              const auto s = static_cast<uint32_t>(cell.strategy);
              Require(s < p.strategies.size() && p.strategies[s].count,
                      "Unsupported strategy");
              const auto q = frame.raw_quant_field().Row(view.block_y +
                                                         y)[view.block_x + x];
              Require(q >= 1 && q <= 256, "Quantizer bound");
              p.anchors.push_back(
                  {anchor_source, g,
                   x | (y << 5) | (s << 10) | (uint32_t(q - 1) << 15), stride});
              anchor_source += p.strategies[s].count;
              ++group.anchors;
              mixed |= s != 0;
            }
          Require(anchor_source - source == stride,
                  "Packed coefficient span differs");
          source += 3 * stride;
        },
        native);
  }
  Require(source == 192 * block_count,
          "Completed packed coefficient length differs");
  p.coefficient_count = source;
  p.scalar = !mixed;
  p.tasks = static_cast<uint32_t>(3 * p.anchors.size());
  p.control = {map.num_contexts,
               static_cast<uint32_t>(map.qf_thresholds.size() + 1), 1, p.tasks};
  p.control.insert(p.control.end(), map.qf_thresholds.begin(),
                   map.qf_thresholds.end());
  p.control.insert(p.control.end(), map.context_map.begin(),
                   map.context_map.end());
  p.end = p.control.size();
  const uint32_t capacity = capacity_hint
                                ? std::min(capacity_hint, p.maximum_tokens)
                                : std::max(1u, (p.maximum_tokens + 7) / 8);
  p.control.insert(p.control.end(),
                   {uint32_t(collect), capacity, 0, 1, uint32_t(p.scalar)});
  return p;
}

}  // namespace

class CudaAcTokenizer final : public AcTokenizationProvider {
 public:
  CudaAcTokenizer(CudaBackend& backend, const DeviceBuffer& coefficients,
                  size_t offset)
      : backend_(backend), coefficients_(coefficients), offset_(offset) {}
  ~CudaAcTokenizer() override {
    // Host count/readback destinations and all device buffers remain alive
    // until the submission is drained, including failed Begin/Finish calls.
    if (submission_) (void)submission_->Wait();
  }
  Status Begin(const Frame& frame, const SimpleCoefficientOrders& orders,
               const SimpleAcNaturalOrders& natural,
               const SimpleBlockContextMap& contexts,
               bool populations) override {
    if (started_)
      return Status::FailedPrecondition("CUDA tokenizer already started");
    started_ = true;
    const resource_budget_internal::ManagedHostScope resources(
        resource_budget_internal::ResourceClass::kSerializer);
    try {
      plan_ = MakePlan(
          frame, orders, natural, contexts, populations,
          backend_.ac_token_capacity_hint_.load(std::memory_order_relaxed));
      Require(offset_ % sizeof(int32_t) == 0 &&
                  offset_ <= coefficients_.size_bytes() &&
                  size_t(plan_.coefficient_count) * sizeof(int32_t) <=
                      coefficients_.size_bytes() - offset_,
              "CUDA completed coefficient buffer is truncated");
      populations_ = populations;
      Allocate(kAnchors, plan_.anchors.size() * sizeof(CudaTokenAnchor));
      Allocate(kGroups, plan_.groups.size() * sizeof(CudaTokenGroup));
      Allocate(kStrategies,
               plan_.strategies.size() * sizeof(CudaTokenStrategy));
      Allocate(kOrders, plan_.orders.size() * sizeof(uint32_t));
      Allocate(kMetadata, size_t(plan_.tasks) * sizeof(CudaTokenMetadata));
      Allocate(kMaps, plan_.map_size);
      Allocate(kHistogram,
               populations ? plan_.context_count * 128 * sizeof(uint32_t) : 4);
      Allocate(kControl, plan_.control.size() * sizeof(uint32_t));
      AllocateOutput(plan_.control[plan_.end + 1]);
      const std::array<CudaHostToDeviceCopy, 5> uploads{
          {{buffers_[kAnchors].get(), plan_.anchors.data(),
            plan_.anchors.size() * sizeof(CudaTokenAnchor), 0},
           {buffers_[kGroups].get(), plan_.groups.data(),
            plan_.groups.size() * sizeof(CudaTokenGroup), 0},
           {buffers_[kStrategies].get(), plan_.strategies.data(),
            plan_.strategies.size() * sizeof(CudaTokenStrategy), 0},
           {buffers_[kOrders].get(), plan_.orders.data(),
            plan_.orders.size() * sizeof(uint32_t), 0},
           {buffers_[kControl].get(), plan_.control.data(),
            plan_.control.size() * sizeof(uint32_t), 0}}};
      Check(backend_.CopyHostToDeviceBatch(uploads));
      Check(backend_.SubmitCompute(&Encode, this, &submission_));
      Require(submission_ != nullptr, "CUDA tokenizer returned no submission");
      ++cuda_token_provider_begin_count;
      return Status::Ok();
    } catch (const Failure& failure) {
      return failure.status;
    } catch (
        const resource_budget_internal::ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("CUDA token preparation allocation failed");
    } catch (const std::length_error&) {
      return Status::OutOfMemory("CUDA token preparation size overflow");
    }
  }
  Status Finish(Storage<EntropyTokenStreamView>* streams,
                Storage<PreparedFixedAnsCluster>* populations) override {
    if (!streams || !populations || !submission_ || finished_ || consumed_)
      return Status::FailedPrecondition(
          "CUDA token result is unavailable or consumed");
    consumed_ = true;
    const resource_budget_internal::ManagedHostScope resources(
        resource_budget_internal::ResourceClass::kSerializer);
    try {
      Check(submission_->Wait());
      const std::array<CudaDeviceToHostCopy, 2> count_reads{
          {{buffers_[kGroups].get(), plan_.groups.data(),
            plan_.groups.size() * sizeof(CudaTokenGroup), 0},
           {buffers_[kControl].get(), plan_.control.data(),
            plan_.control.size() * sizeof(uint32_t), 0}}};
      Check(backend_.CopyDeviceToHostBatch(count_reads));
      const uint32_t total = plan_.control[plan_.end + 2];
      Require(total && total <= plan_.maximum_tokens,
              "CUDA token total exceeds bound");
      uint64_t offset = 0;
      for (const auto& group : plan_.groups) {
        Require(group.output == offset &&
                    group.tokens <= 195u * group.width * group.height,
                "CUDA token group prefix/count differs");
        offset += group.tokens;
      }
      Require(offset == total, "CUDA token total disagrees with group counts");
      if (total > plan_.control[plan_.end + 1]) {
        submission_.reset();
        AllocateOutput(total);
        plan_.control[plan_.end + 1] = total;
        Check(backend_.CopyHostToDevice(
            *buffers_[kControl], plan_.control.data(),
            plan_.control.size() * sizeof(uint32_t), 0));
        Check(backend_.SubmitCompute(&Emit, this, &submission_));
        Require(submission_ != nullptr,
                "CUDA token retry returned no submission");
        Check(submission_->Wait());
        ++cuda_token_provider_retry_count;
      }
      values_.ResetForOverwrite(total);
      contexts_.ResetForOverwrite(total);
      histogram_.ResetForOverwrite(populations_ ? plan_.context_count * 128
                                                : 0);
      std::array<CudaDeviceToHostCopy, 3> reads{
          {{buffers_[kValues].get(), values_.data(), size_t(total) * 4, 0},
           {buffers_[kContexts].get(), contexts_.data(), size_t(total) * 2, 0},
           {buffers_[kHistogram].get(), histogram_.data(),
            histogram_.size() * 4, 0}}};
      Check(backend_.CopyDeviceToHostBatch(
          std::span(reads).first(populations_ ? 3 : 2)));
      Storage<EntropyTokenStreamView> candidate;
      candidate.reserve(plan_.groups.size());
      for (const auto& group : plan_.groups)
        candidate.push_back(EntropyTokenStreamView::Split(
            {values_.data() + group.output, group.tokens},
            {contexts_.data() + group.output, group.tokens}));
      Storage<PreparedFixedAnsCluster> counts(populations_ ? plan_.context_count
                                                           : 0);
      for (size_t c = 0; c < counts.size(); ++c)
        for (uint32_t symbol = 0; symbol < 128; ++symbol) {
          const uint64_t n = histogram_.data()[c * 128 + symbol];
          auto& dst = counts[c];
          dst.counts[symbol] = n;
          dst.token_count += n;
          if (n) dst.maximum_symbol = symbol;
          if (symbol >= 16) dst.extra_bits += n * (2 + ((symbol - 16) >> 2));
        }
      const uint32_t observed_capacity = plan_.control[plan_.end + 1];
      uint32_t previous =
          backend_.ac_token_capacity_hint_.load(std::memory_order_relaxed);
      while (previous < observed_capacity &&
             !backend_.ac_token_capacity_hint_.compare_exchange_weak(
                 previous, observed_capacity, std::memory_order_relaxed)) {
      }
      *streams = std::move(candidate);
      *populations = std::move(counts);
      finished_ = true;
      return Status::Ok();
    } catch (const Failure& failure) {
      return failure.status;
    } catch (
        const resource_budget_internal::ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("CUDA token output allocation failed");
    } catch (const std::length_error&) {
      return Status::OutOfMemory("CUDA token output size overflow");
    }
  }

 private:
  enum Slot {
    kAnchors,
    kGroups,
    kStrategies,
    kOrders,
    kMetadata,
    kMaps,
    kValues,
    kContexts,
    kHistogram,
    kControl,
    kBufferCount
  };
  void Allocate(Slot slot, size_t bytes) {
    std::unique_ptr<DeviceBuffer> replacement;
    Check(backend_.Allocate(std::max(size_t{4}, bytes), &replacement));
    buffers_[slot] = std::move(replacement);
  }
  void AllocateOutput(uint32_t tokens) {
    Allocate(kValues, size_t(tokens) * 4);
    Allocate(kContexts, size_t(tokens) * 2);
  }
  template <class T>
  T* Data(Slot slot) {
    return static_cast<T*>(
        CudaBackend::AsCudaBuffer(*buffers_[slot])->pointer());
  }
  CudaTokenPointers Pointers() {
    return {reinterpret_cast<const int32_t*>(
                static_cast<const std::byte*>(
                    CudaBackend::AsCudaBuffer(coefficients_)->pointer()) +
                offset_),
            Data<CudaTokenAnchor>(kAnchors),
            Data<CudaTokenGroup>(kGroups),
            Data<CudaTokenStrategy>(kStrategies),
            Data<uint32_t>(kOrders),
            Data<CudaTokenMetadata>(kMetadata),
            Data<uint8_t>(kMaps),
            Data<uint32_t>(kValues),
            Data<uint16_t>(kContexts),
            Data<uint32_t>(kHistogram),
            Data<uint32_t>(kControl)};
  }
  static cudaError_t Encode(CudaBackend& backend, const void* opaque) {
    auto& self = *const_cast<CudaAcTokenizer*>(
        static_cast<const CudaAcTokenizer*>(opaque));
    const auto stream = backend.state_->stream;
    auto status =
        cudaMemsetAsync(self.Data<void>(kHistogram), 0,
                        self.buffers_[kHistogram]->size_bytes(), stream);
    if (status != cudaSuccess) return status;
    return LaunchCudaAcTokenization(
        self.Pointers(), self.plan_.tasks,
        static_cast<uint32_t>(self.plan_.groups.size()), self.plan_.scalar,
        stream);
  }
  static cudaError_t Emit(CudaBackend& backend, const void* opaque) {
    auto& self = *const_cast<CudaAcTokenizer*>(
        static_cast<const CudaAcTokenizer*>(opaque));
    return LaunchCudaAcTokenEmit(self.Pointers(), self.plan_.tasks,
                                 self.plan_.scalar, backend.state_->stream);
  }
  CudaBackend& backend_;
  const DeviceBuffer& coefficients_;
  size_t offset_;
  Plan plan_;
  std::array<std::unique_ptr<DeviceBuffer>, kBufferCount> buffers_;
  OverwriteArray<uint32_t> values_, histogram_;
  OverwriteArray<uint16_t> contexts_;
  std::unique_ptr<GpuSubmission> submission_;
  bool populations_ = false, started_ = false, consumed_ = false,
       finished_ = false;
};

Status CreateCudaAcTokenizationProvider(
    GpuBackend& backend, const DeviceBuffer& coefficients, size_t offset,
    std::unique_ptr<AcTokenizationProvider>* out) {
  auto* cuda = dynamic_cast<CudaBackend*>(&backend);
  if (!out || !cuda || coefficients.backend() != BackendKind::kCuda ||
      coefficients.backend_id() != backend.id())
    return Status::InvalidArgument(
        "CUDA token provider backend/output mismatch");
  try {
    *out = std::make_unique<CudaAcTokenizer>(*cuda, coefficients, offset);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("CUDA token provider allocation failed");
  }
}

Status ComputeCudaTokenStoragePlan(
    Extent2D source, resource_budget_internal::HostStorageBound* out) {
  using resource_budget_internal::HostStorageBound;
  using enum resource_budget_internal::VectorCapacityPolicy;
  if (!out) return Status::InvalidArgument("CUDA token storage output is null");
  FrameGeometry geometry;
  auto status = FrameGeometry::Create(source, &geometry);
  if (!status.ok()) return status;
  const auto extent = geometry.block_grid().blocks;
  size_t blocks = 0;
  if (!extent.try_area(&blocks) || blocks > UINT32_MAX / 195u)
    return Status::InvalidArgument("CUDA token geometry exceeds index bound");
  BlockContextMapStoragePlan maps;
  status = ComputeBlockContextMapStoragePlan(extent, false, &maps);
  if (!status.ok()) return status;
  const size_t groups =
      ((extent.width + 31) / 32) * ((extent.height + 31) / 32);
  const size_t tokens = 195 * blocks;
  const size_t control = 9 + maps.maximum_thresholds + maps.maximum_map_entries;
  HostStorageBound device, host;
  const auto add_device = [&](size_t bytes) {
    bytes = std::max(size_t{4}, bytes);
    return device.Add({bytes, bytes});
  };
  if (!add_device(blocks * sizeof(CudaTokenAnchor)) ||
      !add_device(groups * sizeof(CudaTokenGroup)) ||
      !add_device(kAcStrategyCount * sizeof(CudaTokenStrategy)) ||
      !add_device(3 * 2624 * sizeof(uint32_t)) ||
      !add_device(3 * blocks * sizeof(CudaTokenMetadata)) ||
      !add_device(3 * blocks) || !add_device(tokens * 4) ||
      !add_device(tokens * 2) ||
      !add_device(maps.maximum_ac_contexts * 128 * sizeof(uint32_t)) ||
      !add_device(control * sizeof(uint32_t)) ||
      !host.AddVector<CudaTokenAnchor>(blocks, kFreshExact) ||
      !host.AddVector<CudaTokenGroup>(groups, kFreshExact) ||
      !host.AddVector<CudaTokenStrategy>(kAcStrategyCount, kFreshExact) ||
      !host.AddVector<uint32_t>(3 * 2624, kFreshExact) ||
      !host.AddVector<uint32_t>(control, kFreshExact) ||
      !host.AddVector<uint32_t>(tokens, kFreshExact) ||
      !host.AddVector<uint16_t>(tokens, kFreshExact) ||
      !host.AddVector<uint32_t>(maps.maximum_ac_contexts * 128, kFreshExact) ||
      !host.AddVector<PreparedFixedAnsCluster>(maps.maximum_ac_contexts,
                                               kFreshExact) ||
      !host.AddVector<EntropyTokenStreamView>(groups, kFreshExact) ||
      // Cover retry replacement and idle backing from the undersized output.
      // Counting every device owner twice is deliberately conservative.
      !host.Add(device, 2))
    return Status::OutOfMemory("CUDA token storage bound overflows");
  *out = host;
  return Status::Ok();
}
}  // namespace gjxl::cuda_internal
