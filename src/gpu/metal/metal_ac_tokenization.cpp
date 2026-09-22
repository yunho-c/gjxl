// SPDX-License-Identifier: Apache-2.0
#include "codestream/ac_tokenization_provider_internal.h"
#include "codestream/simple_ac_context.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/metal/metal_status.h"
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>

namespace gjxl::metal_internal {
using namespace codestream_internal;
struct TokenAnchor {
  uint32_t source, group, packed, reserved;
};
struct TokenGroup {
  uint32_t first, anchors, output, map, width, height, tokens, reserved;
};
struct TokenStrategy {
  uint32_t count, covered, width, height, order[3], log2covered;
};

class MetalAcTokenizer final : public AcTokenizationProvider {
public:
  MetalAcTokenizer(MetalBackend &backend, const DeviceBuffer &coefficients,
                   size_t offset)
      : backend_(backend), coefficients_(coefficients),
        coefficient_offset_(offset) {}
  ~MetalAcTokenizer() override {
    const bool completed = !submission_ || submission_->Wait().ok();
    backend_.ReleaseAqScratchArena(MetalAqScratchArena::kAcTokenization,
                                   std::move(arena_),
                                   cache_ && finished_ && completed);
    backend_.ReleaseAqScratchArena(MetalAqScratchArena::kAcTokenOutput,
                                   std::move(output_arena_),
                                   cache_ && finished_ && completed);
  }

  Status Initialize() {
    std::lock_guard lock(backend_.ac_tokenization_mutex_);
    constexpr const char *names[] = {
        "gjxl_ac_token_metadata", "gjxl_ac_token_offsets", "gjxl_ac_token_emit",
        "gjxl_ac_token_group_offsets", "gjxl_ac_token_emit_scalar_dct8"};
    for (size_t i = 0; i < 5; ++i) {
      auto &pipeline = backend_.ac_tokenization_pipelines_[i];
      if (!pipeline) {
        auto function = NS::TransferPtr(backend_.library_->newFunction(
            NS::String::string(names[i], NS::UTF8StringEncoding)));
        if (!function)
          return Status::Internal("Missing experimental AC token kernel");
        NS::Error *error = nullptr;
        pipeline = NS::TransferPtr(
            backend_.device_->newComputePipelineState(function.get(), &error));
        if (!pipeline)
          return metal::ErrorToStatus(error, "AC token pipeline");
        RegisterMetalComputePipeline(pipeline.get(), names[i]);
      }
      if (pipeline->threadExecutionWidth() != 32)
        return Status::Unavailable("AC tokenizer requires 32-wide SIMD groups");
      pipelines_[i] = pipeline;
    }
    return Status::Ok();
  }

  Status Begin(const vardct_frame_internal::VarDctFrameView &frame,
               const SimpleCoefficientOrders &orders,
               const SimpleAcNaturalOrders &natural,
               const SimpleBlockContextMap &contexts,
               bool populations) override {
    diagnostic_ = std::getenv("GJXL_EXPERIMENT_TOKEN_DIAGNOSTIC") != nullptr;
    Stamp(0);
    if (submission_)
      return Status::FailedPrecondition("AC tokenizer already submitted");
    const auto blocks = frame.geometry().block_grid().blocks;
    const size_t block_count = blocks.width * blocks.height;
    if (block_count > std::numeric_limits<uint32_t>::max() / 195u ||
        frame.ac_group_count() > std::numeric_limits<uint32_t>::max() / 196608u)
      return Status::InvalidArgument(
          "Experimental AC tokenizer index bound exceeded");
    group_count_ = frame.ac_group_count();
    if (coefficient_offset_ % alignof(int32_t) != 0 ||
        coefficient_offset_ > coefficients_.size_bytes() ||
        group_count_ * size_t{196608} * sizeof(int32_t) >
            coefficients_.size_bytes() - coefficient_offset_)
      return Status::InvalidArgument(
          "Resident AC coefficient buffer is truncated");
    context_count_ = contexts.ac_context_count();
    populations_ = populations;
    compact_ = false;
    if (const char *value = std::getenv("GJXL_EXPERIMENT_TOKEN_COMPACT")) {
      compact_ = value[0] == '1' || value[0] == '2';
      smart_ = value[0] == '2';
    }
    shards_ = 4;
    if (const char *value = std::getenv("GJXL_EXPERIMENT_TOKEN_SHARDS")) {
      const long n = std::strtol(value, nullptr, 10);
      if (n == 1 || n == 2 || n == 4 || n == 8)
        shards_ = static_cast<uint32_t>(n);
    }
    scalar_dct8_ = std::getenv("GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8") != nullptr;
    threads_ = 128;
    if (const char *value = std::getenv("GJXL_EXPERIMENT_TOKEN_THREADS")) {
      const long n = std::strtol(value, nullptr, 10);
      if (n == 64 || n == 128 || n == 256)
        threads_ = static_cast<uint32_t>(n);
    }
    if (threads_ > pipelines_[0]->maxTotalThreadsPerThreadgroup() ||
        threads_ > pipelines_[2]->maxTotalThreadsPerThreadgroup())
      return Status::Unavailable(
          "Token threadgroup width exceeds pipeline limit");
    Status status;
    size_t capacity = 0;
    const auto allocate = [&](size_t index, size_t bytes) -> Status {
      offsets_[index] = (capacity + 255) & ~size_t{255};
      sizes_[index] = std::max(size_t{4}, bytes);
      capacity = offsets_[index] + sizes_[index];
      return Status::Ok();
    };
    // Allocations are admitted to the experimental serializer envelope. Only
    // used output elements are touched; unused group capacity stays untouched.
    const size_t max_tokens = 195 * block_count;
    const size_t segments = contexts.qf_thresholds.size() + 1;
    if (!(status = allocate(0, block_count * sizeof(TokenAnchor))).ok() ||
        !(status = allocate(1, group_count_ * sizeof(TokenGroup))).ok() ||
        !(status = allocate(2, kAcStrategyCount * sizeof(TokenStrategy)))
             .ok() ||
        !(status = allocate(3, 3 * 2624 * sizeof(uint32_t))).ok() ||
        !(status = allocate(4, 3 * block_count * 4 * sizeof(uint32_t))).ok() ||
        !(status = allocate(5, 3 * block_count)).ok() ||
        !(status = allocate(6, compact_ ? 4 : max_tokens * sizeof(uint32_t)))
             .ok() ||
        !(status = allocate(7, compact_ ? 4 : max_tokens * sizeof(uint16_t)))
             .ok() ||
        !(status = allocate(8, populations ? size_t(shards_) * context_count_ *
                                                 128 * sizeof(uint32_t)
                                           : 4))
             .ok() ||
        !(status =
              allocate(9, (4 + segments - 1 + contexts.context_map.size() + 5) *
                              sizeof(uint32_t)))
             .ok())
      return status;

    cache_ = true;
    if (const char *value = std::getenv("GJXL_EXPERIMENT_TOKEN_CACHE"))
      cache_ = value[0] != '0';
    if (cache_)
      status = backend_.AcquireAqScratchArena(
          MetalAqScratchArena::kAcTokenization, capacity, &arena_);
    else
      status = arena_.Prepare(backend_, capacity);
    if (!status.ok())
      return status;
    arena_buffer_ = MetalBackend::AsMetalBuffer(*arena_.backing_buffer());
    Stamp(1);
    auto *strategy_data = Data<TokenStrategy>(2);
    std::memset(strategy_data, 0, kAcStrategyCount * sizeof(TokenStrategy));
    auto *order_data = Data<uint32_t>(3);
    uint32_t order_used = 0;
    for (size_t index = 0; index < kAcStrategyCount; ++index) {
      if (natural.orders[index].empty())
        continue;
      const auto &info = *GetAcStrategyInfo(static_cast<AcStrategyType>(index));
      auto &s = strategy_data[index];
      s.count = static_cast<uint32_t>(info.coefficient_count());
      s.width = static_cast<uint32_t>(info.covered_blocks.width);
      s.height = static_cast<uint32_t>(info.covered_blocks.height);
      s.covered = s.width * s.height;
      s.log2covered = std::countr_zero(s.covered);
      const size_t family = kSimpleStrategyOrder[index];
      for (size_t channel = 0; channel < 3; ++channel) {
        const auto &scan = (orders.used_order_mask & (uint16_t{1} << family))
                               ? orders.orders[family][channel]
                               : natural.orders[index];
        if (scan.size() != s.count || scan.size() > 3 * 2624 - order_used)
          return Status::Internal("Experimental AC order storage mismatch");
        s.order[channel] = order_used;
        std::copy(scan.begin(), scan.end(), order_data + order_used);
        order_used += static_cast<uint32_t>(scan.size());
      }
    }
    auto *control = Data<uint32_t>(9);
    control[0] = contexts.num_contexts;
    control[1] = static_cast<uint32_t>(segments);
    control[2] = shards_;
    std::copy(contexts.qf_thresholds.begin(), contexts.qf_thresholds.end(),
              control + 4);
    std::copy(contexts.context_map.begin(), contexts.context_map.end(),
              control + 4 + segments - 1);
    control_end_ = 4 + segments - 1 + contexts.context_map.size();
    control[control_end_] = populations;
    control[control_end_ + 1] = 0;
    control[control_end_ + 2] = 0;
    control[control_end_ + 3] = smart_;
    control[control_end_ + 4] = scalar_dct8_;
    if (smart_) {
      uint32_t hint = backend_.ac_tokenization_capacity_hint_.load(
          std::memory_order_relaxed);
      if (hint == 0)
        hint = static_cast<uint32_t>((max_tokens + 7) / 8);
      if (const char *forced = std::getenv("GJXL_EXPERIMENT_TOKEN_CAPACITY"))
        hint = std::max(1ul, std::strtoul(forced, nullptr, 10));
      status = PrepareOutput(
          static_cast<uint32_t>(std::min<size_t>(hint, max_tokens)));
      if (!status.ok())
        return status;
    }
    if (populations)
      std::memset(Data<uint32_t>(8), 0, sizes_[8]);
    auto *anchors = Data<TokenAnchor>(0);
    auto *groups = Data<TokenGroup>(1);
    const auto quant = frame.raw_quant_field();
    uint32_t anchor_count = 0, output = 0, map = 0;
    for (size_t group = 0; group < group_count_; ++group) {
      VarDctAcGroupView view;
      if (!(status = frame.GetAcGroup(group, &view)).ok())
        return status;
      auto &g = groups[group];
      g = {anchor_count,
           0,
           output,
           map,
           static_cast<uint32_t>(view.block_extent.width),
           static_cast<uint32_t>(view.block_extent.height),
           0,
           0};
      uint32_t source = static_cast<uint32_t>(group * 196608);
      for (size_t y = 0; y < view.block_extent.height; ++y)
        for (size_t x = 0; x < view.block_extent.width; ++x) {
          AcStrategyCell cell;
          if (!(status = frame.strategies().Get(view.block_x + x,
                                                view.block_y + y, &cell))
                   .ok())
            return status;
          if (!cell.is_anchor)
            continue;
          const uint32_t strategy = static_cast<uint32_t>(cell.strategy);
          has_dct8_ |= strategy == 0;
          has_other_ |= strategy != 0;
          if (strategy_data[strategy].count == 0)
            return Status::InvalidArgument(
                "Unsupported experimental token transform");
          const int32_t raw_quant =
              quant.Row(view.block_y + y)[view.block_x + x];
          if (raw_quant < 1 || raw_quant > 256)
            return Status::InvalidArgument(
                "Invalid experimental token quantizer");
          anchors[anchor_count++] = {
              source, static_cast<uint32_t>(group),
              static_cast<uint32_t>(x | (y << 5) | (strategy << 10) |
                                    (uint32_t(raw_quant - 1) << 15)),
              0};
          source += strategy_data[strategy].count;
          ++g.anchors;
        }
      output += 3 * (64 * g.width * g.height + g.anchors);
      map += 3 * g.width * g.height;
    }
    // Auto scalar mode is limited to a pure DCT8 frame; mixed strategy screens
    // did not improve. The explicit value 1 retains the mixed-frame ablation.
    if (const char *scalar = std::getenv("GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8")) {
      if (scalar[0] == '2' && has_other_)
        scalar_dct8_ = false;
    }
    control[control_end_ + 4] = scalar_dct8_;
    const char *group_mode = std::getenv("GJXL_EXPERIMENT_TOKEN_GROUP_DCT8");
    group_dct8_ = smart_ && !has_other_ && group_mode && group_mode[0] == '1';
    group_threads_ = threads_;
    if (const char *value =
            std::getenv("GJXL_EXPERIMENT_TOKEN_GROUP_THREADS")) {
      const long n = std::strtol(value, nullptr, 10);
      if (n == 64 || n == 128 || n == 256)
        group_threads_ = uint32_t(n);
    }
    if (group_dct8_) {
      std::lock_guard lock(backend_.ac_tokenization_mutex_);
      auto &pipeline = backend_.ac_tokenization_pipelines_[5];
      constexpr const char *name = "gjxl_ac_token_group_dct8";
      if (!pipeline) {
        auto function = NS::TransferPtr(backend_.library_->newFunction(
            NS::String::string(name, NS::UTF8StringEncoding)));
        if (!function)
          return Status::Internal("Missing grouped DCT8 tokenizer");
        NS::Error *error = nullptr;
        pipeline = NS::TransferPtr(
            backend_.device_->newComputePipelineState(function.get(), &error));
        if (!pipeline)
          return metal::ErrorToStatus(error, "Grouped DCT8 pipeline");
        RegisterMetalComputePipeline(pipeline.get(), name);
      }
      if (pipeline->threadExecutionWidth() != 32 ||
          group_threads_ > pipeline->maxTotalThreadsPerThreadgroup())
        return Status::Unavailable("Grouped DCT8 threadgroup is unsupported");
      pipelines_[5] = pipeline;
    }
    tasks_ = 3 * anchor_count;
    if (diagnostic_)
      std::fprintf(stderr,
                   "TOKEN_SETUP anchors=%u contexts=%zu shards=%u capacity=%zu "
                   "cache=%d\n",
                   anchor_count, context_count_, shards_, capacity, cache_);
    control[3] = tasks_;
    Stamp(2);
    kernel_profile_ =
        std::getenv("GJXL_EXPERIMENT_TOKEN_KERNEL_PROFILE") != nullptr;
    if (kernel_profile_) {
      const std::array stages = {
          MetalProfiledComputeStage{.stage_id = "token.metadata",
                                    .group_id = "token",
                                    .encode = EncodeStage<0>,
                                    .context = this},
          MetalProfiledComputeStage{.stage_id = "token.offsets",
                                    .group_id = "token",
                                    .encode = EncodeStage<1>,
                                    .context = this},
          MetalProfiledComputeStage{.stage_id = "token.group_offsets",
                                    .group_id = "token",
                                    .encode = EncodeStage<3>,
                                    .context = this},
          MetalProfiledComputeStage{.stage_id = "token.emit",
                                    .group_id = "token",
                                    .encode = EncodeStage<2>,
                                    .context = this}};
      if (group_dct8_) {
        const MetalProfiledComputeStage stage{.stage_id = "token.group_dct8",
                                              .group_id = "token",
                                              .encode = EncodeStage<5>,
                                              .context = this};
        status = backend_.SubmitComputeProfiled(
            "Grouped DCT8 tokenization", {&stage, 1},
            gpu_profile_internal::GpuProfilingMode::kStage, &submission_);
      } else if (smart_)
        status = backend_.SubmitComputeProfiled(
            "AC tokenization", stages,
            gpu_profile_internal::GpuProfilingMode::kStage, &submission_);
      else if (compact_)
        status = backend_.SubmitComputeProfiled(
            "AC tokenization", std::span(stages).first(2),
            gpu_profile_internal::GpuProfilingMode::kStage, &submission_);
      else {
        const std::array normal = {stages[0], stages[1], stages[3]};
        status = backend_.SubmitComputeProfiled(
            "AC tokenization", normal,
            gpu_profile_internal::GpuProfilingMode::kStage, &submission_);
      }
    } else {
      status =
          backend_.SubmitCompute("AC tokenization", Encode, this, &submission_);
    }
    Stamp(3);
    return status;
  }

  Status Finish(Storage<EntropyTokenStreamView> *streams,
                Storage<PreparedFixedAnsCluster> *populations) override {
    if (!submission_ || !streams || !populations)
      return Status::InvalidArgument("Missing AC token submission/output");
    if (finished_)
      return Status::FailedPrecondition("AC token output already consumed");
    Stamp(4);
    auto status = submission_->Wait();
    Stamp(5);
    if (!status.ok())
      return status;
    status = LogSubmission();
    if (!status.ok())
      return status;
    if (compact_) {
      auto *groups = Data<TokenGroup>(1);
      uint32_t total = 0;
      for (size_t i = 0; i < group_count_; ++i) {
        auto &g = groups[i];
        if (g.tokens > 3 * (64 * g.width * g.height + g.anchors))
          return Status::Internal("GPU compact group token bound exceeded");
        if (!group_dct8_)
          g.output = total;
        total += g.tokens;
      }
      if (group_dct8_ && Data<uint32_t>(9)[control_end_ + 2] != total)
        return Status::Internal("Grouped DCT8 total differs from group counts");
      if (smart_)
        backend_.ac_tokenization_capacity_hint_.store(
            std::max(total, output_capacity_tokens_),
            std::memory_order_relaxed);
      if (!smart_ || total > output_capacity_tokens_) {
        submission_.reset(); // The first command has completed; release its
                             // references.
        status = PrepareOutput(total);
        if (!status.ok())
          return status;
        if (group_dct8_) {
          Data<uint32_t>(9)[control_end_ + 2] = 0;
          if (populations_)
            std::memset(Data<uint32_t>(8), 0, sizes_[8]);
        }
        if (diagnostic_)
          std::fprintf(stderr,
                       "TOKEN_COMPACT output_capacity=%zu fallback=%d\n",
                       output_arena_.capacity_bytes(), smart_);
        if (kernel_profile_) {
          const MetalProfiledComputeStage stage{
              .stage_id = group_dct8_ ? "token.group_dct8" : "token.emit",
              .group_id = "token",
              .encode = group_dct8_ ? EncodeStage<5> : EncodeStage<2>,
              .context = this};
          status = backend_.SubmitComputeProfiled(
              "AC token emit", {&stage, 1},
              gpu_profile_internal::GpuProfilingMode::kStage, &submission_);
        } else
          status = backend_.SubmitCompute(
              "AC token emit", group_dct8_ ? EncodeStage<5> : EncodeStage<2>,
              this, &submission_);
        if (!status.ok())
          return status;
        status = submission_->Wait();
        if (!status.ok())
          return status;
        status = LogSubmission();
        if (!status.ok())
          return status;
        Stamp(5); // Compact wait includes sizing and its second GPU submission.
      }
    }
    Storage<EntropyTokenStreamView> result;
    result.reserve(group_count_);
    const auto *groups = Data<TokenGroup>(1);
    const auto *values = Data<uint32_t>(6);
    const auto *contexts = Data<uint16_t>(7);
    uint64_t total_tokens = 0;
    for (size_t i = 0; i < group_count_; ++i) {
      const auto &g = groups[i];
      if (g.tokens > 3 * (64 * g.width * g.height + g.anchors))
        return Status::Internal("GPU token output exceeds group bound");
      if (group_dct8_ && (g.output > output_capacity_tokens_ ||
                          g.tokens > output_capacity_tokens_ - g.output))
        return Status::Internal("Grouped DCT8 output exceeds capacity");
      result.push_back(EntropyTokenStreamView::Split(
          {values + g.output, g.tokens}, {contexts + g.output, g.tokens}));
      total_tokens += g.tokens;
    }
    Storage<PreparedFixedAnsCluster> reduced;
    if (populations_) {
      reduced.resize(context_count_);
      const auto *hist = Data<uint32_t>(8);
      uint64_t population_tokens = 0;
      if (std::getenv("GJXL_EXPERIMENT_TOKEN_SPECIALIZED_REDUCE")) {
        switch (shards_) {
        case 1:
          population_tokens = ReduceHistograms<1>(hist, &reduced);
          break;
        case 2:
          population_tokens = ReduceHistograms<2>(hist, &reduced);
          break;
        case 4:
          population_tokens = ReduceHistograms<4>(hist, &reduced);
          break;
        case 8:
          population_tokens = ReduceHistograms<8>(hist, &reduced);
          break;
        }
      } else {
        for (size_t context = 0; context < context_count_; ++context) {
          auto &dst = reduced[context];
          for (size_t symbol = 0; symbol < 128; ++symbol) {
            uint64_t count = 0;
            for (size_t shard = 0; shard < shards_; ++shard)
              count += hist[(shard * context_count_ + context) * 128 + symbol];
            dst.counts[symbol] = count;
            dst.token_count += count;
            if (count)
              dst.maximum_symbol = static_cast<uint32_t>(symbol);
            if (symbol >= 16)
              dst.extra_bits += count * (2 + ((symbol - 16) >> 2));
          }
          population_tokens += dst.token_count;
        }
      }
      if (population_tokens != total_tokens)
        return Status::Internal(
            "GPU token populations differ from token count");
    }
    *streams = std::move(result);
    *populations = std::move(reduced);
    finished_ = true;
    Stamp(6);
    if (diagnostic_)
      std::fprintf(stderr,
                   "TOKEN_DIAGNOSTIC alloc=%.3f prepare=%.3f submit=%.3f "
                   "gap=%.3f wait=%.3f reduce=%.3f tokens=%llu\n",
                   Millis(0, 1), Millis(1, 2), Millis(2, 3), Millis(3, 4),
                   Millis(4, 5), Millis(5, 6),
                   static_cast<unsigned long long>(total_tokens));
    return Status::Ok();
  }

private:
  template <size_t shards>
  uint64_t ReduceHistograms(const uint32_t *hist,
                            Storage<PreparedFixedAnsCluster> *out) {
    uint64_t total = 0;
    for (size_t context = 0; context < context_count_; ++context) {
      auto &dst = (*out)[context];
      for (size_t symbol = 0; symbol < 128; ++symbol) {
        uint64_t count = 0;
        for (size_t shard = 0; shard < shards; ++shard)
          count += hist[(shard * context_count_ + context) * 128 + symbol];
        dst.counts[symbol] = count;
        dst.token_count += count;
        if (count)
          dst.maximum_symbol = static_cast<uint32_t>(symbol);
        if (symbol >= 16)
          dst.extra_bits += count * (2 + ((symbol - 16) >> 2));
      }
      total += dst.token_count;
    }
    return total;
  }
  Status PrepareOutput(uint32_t tokens) {
    // Previous speculative output is no longer in use: Finish waited first.
    output_arena_ = DeviceScratchArena{};
    output_capacity_tokens_ = tokens;
    offsets_[6] = 0;
    offsets_[7] = (size_t(tokens) * sizeof(uint32_t) + 255) & ~size_t{255};
    const size_t capacity = offsets_[7] + size_t(tokens) * sizeof(uint16_t);
    Status status = cache_ ? backend_.AcquireAqScratchArena(
                                 MetalAqScratchArena::kAcTokenOutput, capacity,
                                 &output_arena_)
                           : output_arena_.Prepare(backend_, capacity);
    if (!status.ok())
      return status;
    output_buffer_ =
        MetalBackend::AsMetalBuffer(*output_arena_.backing_buffer());
    Data<uint32_t>(9)[control_end_ + 1] = tokens;
    return Status::Ok();
  }
  Status LogSubmission() {
    Status status;
    if (diagnostic_ && !kernel_profile_) {
      uint64_t ns = 0;
      status = GetMetalSubmissionGpuDuration(*submission_, &ns);
      if (!status.ok())
        return status;
      std::fprintf(stderr, "TOKEN_GPU command=%.3f\n", ns / 1e6);
    }
    if (kernel_profile_) {
      gpu_profile_internal::GpuSubmissionProfile profile;
      status = GetMetalSubmissionGpuProfile(*submission_, &profile);
      if (!status.ok())
        return status;
      std::fprintf(stderr, "TOKEN_GPU command=%.3f",
                   profile.command_buffer_gpu_nanoseconds / 1e6);
      for (const auto &stage : profile.stages)
        std::fprintf(stderr, " %s=%.3f", stage.stage_id.c_str(),
                     stage.gpu_nanoseconds / 1e6);
      std::fprintf(stderr, "\n");
    }
    return Status::Ok();
  }
  void Stamp(size_t i) {
    if (diagnostic_)
      times_[i] = std::chrono::steady_clock::now();
  }
  double Millis(size_t a, size_t b) const {
    return std::chrono::duration<double, std::milli>(times_[b] - times_[a])
        .count();
  }
  bool diagnostic_ = false, kernel_profile_ = false;
  std::array<std::chrono::steady_clock::time_point, 7> times_;
  template <class T> T *Data(size_t i) const {
    return reinterpret_cast<T *>(
        static_cast<std::byte *>(
            ((compact_ && (i == 6 || i == 7)) ? output_buffer_ : arena_buffer_)
                ->handle()
                ->contents()) +
        offsets_[i]);
  }
  static void Encode(MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
                     const void *opaque) {
    const auto &self = *static_cast<const MetalAcTokenizer *>(opaque);
    if (self.group_dct8_) {
      EncodeStage<5>(backend, encoder, opaque);
      return;
    }
    EncodeStage<0>(backend, encoder, opaque);
    EncodeStage<1>(backend, encoder, opaque);
    if (self.smart_)
      EncodeStage<3>(backend, encoder, opaque);
    if (!self.compact_ || self.smart_)
      EncodeStage<2>(backend, encoder, opaque);
  }
  template <size_t stage>
  static void EncodeStage(MetalBackend &, MTL::ComputeCommandEncoder *encoder,
                          const void *opaque) {
    const auto &self = *static_cast<const MetalAcTokenizer *>(opaque);
    const auto bind = [&](size_t slot, size_t buffer) {
      encoder->setBuffer(((self.compact_ && (buffer == 6 || buffer == 7))
                              ? self.output_buffer_
                              : self.arena_buffer_)
                             ->handle(),
                         self.offsets_[buffer], slot);
    };
    if constexpr (stage == 0) {
      encoder->setComputePipelineState(self.pipelines_[0].get());
      RecordMetalComputePipelineState(self.pipelines_[0].get());
      encoder->setBuffer(
          MetalBackend::AsMetalBuffer(self.coefficients_)->handle(),
          self.coefficient_offset_, 0);
      for (size_t i = 0; i < 6; ++i)
        bind(i + 1, i);
      encoder->setBytes(&self.tasks_, sizeof(self.tasks_), 7);
      DispatchMetalThreadgroups(
          encoder,
          MTL::Size((self.tasks_ + self.threads_ / 32 - 1) /
                        (self.threads_ / 32),
                    1, 1),
          MTL::Size(self.threads_, 1, 1));
    } else if constexpr (stage == 1) {
      encoder->setComputePipelineState(self.pipelines_[1].get());
      RecordMetalComputePipelineState(self.pipelines_[1].get());
      bind(0, 1);
      bind(1, 4);
      DispatchMetalThreadgroups(encoder, MTL::Size(self.group_count_, 1, 1),
                                MTL::Size(32, 1, 1));
    } else if constexpr (stage == 3) {
      encoder->setComputePipelineState(self.pipelines_[3].get());
      RecordMetalComputePipelineState(self.pipelines_[3].get());
      bind(0, 1);
      bind(1, 9);
      const uint32_t count = static_cast<uint32_t>(self.group_count_);
      encoder->setBytes(&count, sizeof(count), 2);
      DispatchMetalThreadgroups(encoder, MTL::Size(1, 1, 1),
                                MTL::Size(32, 1, 1));
    } else if constexpr (stage == 2) {
      encoder->setComputePipelineState(self.pipelines_[2].get());
      RecordMetalComputePipelineState(self.pipelines_[2].get());
      encoder->setBuffer(
          MetalBackend::AsMetalBuffer(self.coefficients_)->handle(),
          self.coefficient_offset_, 0);
      for (size_t i = 0; i < 10; ++i)
        bind(i + 1, i);
      if (!self.scalar_dct8_ || self.has_other_)
        DispatchMetalThreadgroups(
            encoder,
            MTL::Size((self.tasks_ + self.threads_ / 32 - 1) /
                          (self.threads_ / 32),
                      1, 1),
            MTL::Size(self.threads_, 1, 1));
      if (self.scalar_dct8_ && self.has_dct8_)
        EncodeStage<4>(self.backend_, encoder, opaque);
    } else if constexpr (stage == 4) {
      encoder->setComputePipelineState(self.pipelines_[4].get());
      RecordMetalComputePipelineState(self.pipelines_[4].get());
      encoder->setBuffer(
          MetalBackend::AsMetalBuffer(self.coefficients_)->handle(),
          self.coefficient_offset_, 0);
      for (size_t i = 0; i < 10; ++i)
        bind(i + 1, i);
      DispatchMetalThreadgroups(
          encoder,
          MTL::Size((self.tasks_ + self.threads_ - 1) / self.threads_, 1, 1),
          MTL::Size(self.threads_, 1, 1));
    } else {
      encoder->setComputePipelineState(self.pipelines_[5].get());
      RecordMetalComputePipelineState(self.pipelines_[5].get());
      encoder->setBuffer(
          MetalBackend::AsMetalBuffer(self.coefficients_)->handle(),
          self.coefficient_offset_, 0);
      for (size_t i = 0; i < 10; ++i)
        bind(i + 1, i);
      DispatchMetalThreadgroups(encoder, MTL::Size(self.group_count_, 1, 1),
                                MTL::Size(self.group_threads_, 1, 1));
    }
  }
  MetalBackend &backend_;
  const DeviceBuffer &coefficients_;
  size_t coefficient_offset_, group_count_ = 0, context_count_ = 0;
  uint32_t shards_ = 4, tasks_ = 0, threads_ = 128, group_threads_ = 128;
  bool populations_ = false, scalar_dct8_ = false, has_dct8_ = false,
       has_other_ = false, group_dct8_ = false;
  DeviceScratchArena arena_, output_arena_;
  MetalBuffer *arena_buffer_ = nullptr;
  MetalBuffer *output_buffer_ = nullptr;
  std::array<size_t, 10> offsets_{}, sizes_{};
  bool cache_ = false, finished_ = false, compact_ = false, smart_ = false;
  uint32_t output_capacity_tokens_ = 0;
  size_t control_end_ = 0;
  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 6> pipelines_;
  std::unique_ptr<GpuSubmission> submission_;
};
} // namespace gjxl::metal_internal

namespace gjxl::codestream_internal {
Status CreateMetalAcTokenizationProvider(
    GpuBackend &backend, const DeviceBuffer &coefficients, size_t offset,
    std::unique_ptr<AcTokenizationProvider> *out) {
  auto *metal = dynamic_cast<metal_internal::MetalBackend *>(&backend);
  if (!metal || !out || !backend.owns(coefficients))
    return Status::InvalidArgument("AC tokenization backend mismatch");
  auto candidate = std::make_unique<metal_internal::MetalAcTokenizer>(
      *metal, coefficients, offset);
  auto status = candidate->Initialize();
  if (!status.ok())
    return status;
  *out = std::move(candidate);
  return Status::Ok();
}
} // namespace gjxl::codestream_internal
