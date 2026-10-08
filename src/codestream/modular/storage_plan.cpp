// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/storage_plan.h"
#include "codestream/headers_internal.h"
#include "codestream/modular/tree_codec.h"
#include "codestream/sections.h"
#include <algorithm>

namespace gjxl::modular_internal {
using namespace codestream_internal;
using namespace resource_budget_internal;
namespace {
bool AddBits(size_t n, size_t *total) {
  if (n > SIZE_MAX - *total)
    return false;
  *total += n;
  return true;
}
Status Overflow() { return Status::InvalidArgument("Modular workflow storage overflow"); }
} // namespace
Status ComputeModularWorkflowStoragePlan(Extent2D extent, EntropyCodingMode mode,
                                         ModularWorkflowStoragePlan *out) {
  return ComputeModularWorkflowStoragePlan(extent, PackedModularFormat::kRgb8, mode, out);
}
Status ComputeModularWorkflowStoragePlan(Extent2D extent, PackedModularFormat format,
                                         EntropyCodingMode mode, ModularWorkflowStoragePlan *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular workflow plan");
  if (mode != EntropyCodingMode::kPrefix && mode != EntropyCodingMode::kAns)
    return Status::Unsupported("Unsupported Modular entropy mode");
  ModularWorkflowStoragePlan p;
  if (auto s = ModularFrameGeometry::Create(extent, &p.geometry); !s.ok())
    return s;
  ModularInputProfile profile;
  if (auto s = ResolveModularInput(extent, format, &profile); !s.ok())
    return s;
  const size_t channels = profile.channel_count;
  size_t area;
  if (!extent.try_area(&area) || uint64_t{area} > (uint64_t{1} << 40) || area > SIZE_MAX / channels)
    return Overflow();
  p.tokens = channels * area;
  const auto descriptors = profile.channels();
  ModularStreamStoragePlan layout;
  if (auto s = ComputeModularStreamStoragePlan(p.geometry, descriptors, 0, &layout); !s.ok())
    return s;
  p.streams = layout.streams;
  HostStorageBound image, tokens = layout.owned;
  if (auto s = ComputeModularImageStorageBound(descriptors, 0, &image); !s.ok())
    return s;
  if (!tokens.AddVector<EntropyToken>(p.tokens, VectorCapacityPolicy::kFreshExact) ||
      !tokens.AddVector<EntropyTokenStreamView>(p.streams, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.preparation = tokens;
  if (!p.preparation.Add(image))
    return Overflow();

  EntropyOptimizationStoragePlan prefix, ans;
  const EntropyOptimizationStorageOptions options{.policy = EntropyStoragePolicy::kPrefix,
                                                  .tokens = p.tokens,
                                                  .contexts = 1,
                                                  .sections = p.streams,
                                                  .return_cost = false};
  if (auto s = ComputeEntropyOptimizationStoragePlan(options, &prefix); !s.ok())
    return s;
  HostStorageBound model_work = prefix.working;
  if (mode == EntropyCodingMode::kAns) {
    auto ans_options = options;
    ans_options.policy = EntropyStoragePolicy::kAnsFromPrefix;
    if (auto s = ComputeEntropyOptimizationStoragePlan(ans_options, &ans); !s.ok())
      return s;
    HostStorageBound concurrent = prefix.output;
    if (!concurrent.Add(ans.working))
      return Overflow();
    model_work.peak_bytes = std::max(model_work.peak_bytes, concurrent.peak_bytes);
    model_work.retained_bytes = 0;
  }
  p.modeling = tokens;
  if (!p.modeling.Add(model_work))
    return Overflow();

  GlobalTreeStoragePlan tree;
  EntropyModelStoragePlan model;
  EntropyTokenEmissionStoragePlan payload;
  const bool global = p.geometry.group_count() == 1;
  const size_t maximum_stream_tokens =
      global ? p.tokens : channels * kGroupDimension * kGroupDimension;
  if (auto s = ComputeGlobalTreeStoragePlan(kGradientTreeTokens.size(), &tree); !s.ok())
    return s;
  if (auto s = ComputeEntropyModelStoragePlan(mode, 1, 1, &model); !s.ok())
    return s;
  if (auto s = ComputeEntropyTokenEmissionStoragePlan(mode, maximum_stream_tokens, &payload);
      !s.ok())
    return s;
  // default DC matrices + tree + no-LZ77/model + stream header, then padding.
  p.maximum_global_bits = 1;
  if (!AddBits(tree.maximum_bits, &p.maximum_global_bits) ||
      !AddBits(1 + model.maximum_bits, &p.maximum_global_bits) ||
      !AddBits(4 + 7, &p.maximum_global_bits) ||
      (global && !AddBits(payload.maximum_bits, &p.maximum_global_bits)))
    return Overflow();
  p.maximum_group_bits = global ? 0 : payload.maximum_bits;
  if (!global && !AddBits(4 + 7, &p.maximum_group_bits))
    return Overflow();
  if (p.maximum_global_bits / 8 + 1 > kMaximumTocSectionSize ||
      p.maximum_group_bits / 8 + 1 > kMaximumTocSectionSize)
    return Status::Unsupported("Modular section bound exceeds the TOC limit");
  HostStorageBound sections, global_writer, group_writer;
  const size_t count = p.geometry.section_count();
  if (!sections.AddVector<BitWriter>(count, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  if (auto s = ComputeEntropyWriterStorageBound(p.maximum_global_bits, &global_writer); !s.ok())
    return s;
  if (!sections.Add(global_writer))
    return Overflow();
  size_t section_bits = p.maximum_global_bits;
  if (!global) {
    if (auto s = ComputeEntropyWriterStorageBound(p.maximum_group_bits, &group_writer); !s.ok())
      return s;
    if (!sections.Add(group_writer, p.geometry.group_count()) ||
        p.maximum_group_bits > SIZE_MAX / p.geometry.group_count() ||
        !AddBits(p.maximum_group_bits * p.geometry.group_count(), &section_bits))
      return Overflow();
  }
  p.emission = tokens;
  if (!p.emission.Add(model.owned) || !p.emission.Add(sections) || !p.emission.Add(tree.scratch) ||
      !p.emission.Add(model.write_scratch) || !p.emission.Add(payload.scratch))
    return Overflow();

  // Assembly retains sections; headers have bounded temporary writers. TOC
  // reserves up to 32 bits per entry plus 15 alignment/flag bits.
  if (count > (SIZE_MAX - 15) / 32)
    return Overflow();
  size_t file_bits = section_bits;
  if (!AddBits(32 * count + 15, &file_bits) || !AddBits(256 + 128, &file_bits))
    return Overflow();
  HostStorageBound file_writer;
  CommonHeaderStoragePlan headers;
  if (auto s = ComputeEntropyWriterStorageBound(file_bits, &file_writer); !s.ok())
    return s;
  if (auto s = ComputeCommonHeaderStoragePlan(0, &headers); !s.ok())
    return s;
  p.assembly = layout.owned;
  if (!p.assembly.Add(sections) || !p.assembly.Add(file_writer) ||
      !p.assembly.Add(headers.temporary) ||
      !p.assembly.AddVector<size_t>(count, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.maximum_codestream_bytes = file_bits / 8 + (file_bits % 8 != 0);
  if (!p.output.AddVector<uint8_t>(p.maximum_codestream_bytes, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.publication = layout.owned;
  if (!p.publication.Add(file_writer) || !p.publication.Add(p.output))
    return Overflow();
  p.working = p.output;
  for (auto phase : {p.preparation, p.modeling, p.emission, p.assembly, p.publication})
    p.working.peak_bytes = std::max(p.working.peak_bytes, phase.peak_bytes);
  *out = p;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
