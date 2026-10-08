// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/workflow.h"
#include "codestream/modular/storage_plan.h"
#include "modular_reference.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/modular/transform/enc_squeeze.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include <map>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
size_t cases = 0, files = 0;
std::filesystem::path artifacts;
void Check(bool b, const char *m) {
  if (!b)
    throw std::runtime_error(m);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void ReferenceStep(jxl::Image &image, const TransformDescriptor &t) {
  if (t.kind == TransformKind::kSqueeze) {
    jxl::SqueezeParams params;
    params.horizontal = t.horizontal;
    params.in_place = t.in_place;
    params.begin_c = t.begin;
    params.num_c = t.count;
    Check(bool(jxl::FwdSqueeze(image, {params}, nullptr)), "Pinned forward squeeze failed");
    return;
  }
  // Prescribed exact palette: independent ordered dictionary and pinned shape application.
  std::map<std::vector<int32_t>, int32_t> dictionary;
  std::vector<std::vector<int32_t>> pixels;
  const size_t w = image.channel[t.begin].w, h = image.channel[t.begin].h;
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      std::vector<int32_t> v;
      for (size_t c = 0; c < t.count; ++c)
        v.push_back(image.channel[t.begin + c].Row(y)[x]);
      dictionary.emplace(v, 0);
      pixels.push_back(std::move(v));
    }
  Check(dictionary.size() <= t.colors, "Bad palette fixture");
  jxl::Transform transform(jxl::TransformId::kPalette);
  transform.begin_c = t.begin;
  transform.num_c = t.count;
  transform.nb_colors = t.colors;
  transform.nb_deltas = 0;
  Check(bool(transform.MetaApply(image)), "Pinned palette shape failed");
  for (size_t c = 0; c < t.count; ++c)
    std::fill_n(image.channel[0].Row(c), t.colors, 0);
  int32_t index = 0;
  for (auto &[color, id] : dictionary) {
    id = index++;
    for (size_t c = 0; c < t.count; ++c)
      image.channel[0].Row(c)[id] = color[c];
  }
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      image.channel[t.begin + 1].Row(y)[x] = dictionary.at(pixels[y * w + x]);
}
void Case(Extent2D extent, PackedModularFormat format, size_t colors, ModularCodingPolicy policy) {
  ModularInputProfile profile;
  Ok(ResolveModularInput(extent, format, &profile));
  ref::IntegerImage expected{extent.width,
                             extent.height,
                             static_cast<uint32_t>(profile.channel_count),
                             profile.metadata.bits,
                             {}};
  const size_t row = extent.width * profile.channel_count * profile.bytes_per_sample,
               stride = row + 3;
  std::vector<uint8_t> bytes((extent.height - 1) * stride + row + 1, 0xcd);
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < profile.channel_count; ++c) {
        const size_t v = colors ? ((x + 7 * y) % colors) : (x * 739 + y * 131 + x * y * 37);
        uint16_t sample = static_cast<uint16_t>((v * (c * 2 + 1) + c * 19) &
                                                (profile.metadata.bits == 8 ? 255 : 65535));
        if (c == 3 && v % 3 == 0)
          sample = 0;
        expected.samples.push_back(sample);
        const size_t i =
            1 + y * stride + (x * profile.channel_count + c) * profile.bytes_per_sample;
        bytes[i] = static_cast<uint8_t>(sample);
        if (profile.bytes_per_sample == 2)
          bytes[i + 1] = static_cast<uint8_t>(sample >> 8);
      }
  const auto unchanged = bytes;
  PackedModularImageView input{std::span(bytes).subspan(1), extent, stride, format};
  ModularEncoderFrame initial, transformed;
  Ok(ModularEncoderFrame::Prepare(input, policy.rct, &initial));
  Ok(ModularEncoderFrame::Prepare(input, policy, &transformed));
  ModularImage restored;
  Ok(InverseTransforms(transformed.image(), profile.channels(), 0, policy.transforms, &restored));
  for (size_t c = 0; c < profile.channel_count; ++c)
    Check(std::ranges::equal(restored.view(c).backing, initial.image().view(c).backing),
          "Native inverse mismatch");
  JxlMemoryManager memory;
  Check(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Pinned allocator failed");
  auto created =
      jxl::Image::Create(&memory, extent.width, extent.height, expected.bits, expected.channels);
  Check(created.ok(), "Pinned image failed");
  auto reference = std::move(created).value_();
  for (size_t c = 0; c < expected.channels; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        reference.channel[c].Row(y)[x] =
            expected.samples[(y * extent.width + x) * expected.channels + c];
  if (policy.rct) {
    auto next =
        jxl::Image::Create(&memory, extent.width, extent.height, expected.bits, expected.channels);
    Check(next.ok(), "Pinned RCT allocation failed");
    auto rct = std::move(next).value_();
    Check(
        bool(jxl::FwdRct({&reference.channel[0], &reference.channel[1], &reference.channel[2]},
                         {&rct.channel[0], &rct.channel[1], &rct.channel[2]}, policy.rct, nullptr)),
        "Pinned RCT failed");
    if (expected.channels == 4)
      rct.channel[3] = std::move(reference.channel[3]);
    reference = std::move(rct);
  }
  TransformSequence prefix;
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    prefix.entries[i] = policy.transforms.entries[i];
    ++prefix.size;
    ReferenceStep(reference, prefix.entries[i]);
    ModularImage native;
    Ok(ForwardTransforms(initial.image(), prefix, &native));
    Check(native.channel_count() == reference.channel.size() &&
              native.metadata_channels() == reference.nb_meta_channels,
          "Transformed channel ordering mismatch");
    for (size_t c = 0; c < native.channel_count(); ++c) {
      const auto v = native.view(c);
      const auto &r = reference.channel[c];
      Check(v.descriptor.extent == Extent2D{r.w, r.h}, "Transform shape mismatch");
      if (c >= native.metadata_channels())
        Check(v.descriptor.hshift == r.hshift && v.descriptor.vshift == r.vshift,
              "Transform shifts mismatch");
      for (size_t y = 0; y < r.h; ++y)
        for (size_t x = 0; x < r.w; ++x)
          Check(v.Row(y)[x] == r.Row(y)[x], "Prescribed transformed sample mismatch");
    }
  }
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularWorkflowStoragePlan plan;
    Ok(ComputeModularWorkflowStoragePlan(extent, format, mode, policy, &plan));
    std::shared_ptr<const ExecutionDomain> domain;
    Ok(ExecutionDomain::Create(
        {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 1}, &domain));
    std::vector<uint8_t> output;
    Ok(EncodeModularImage(input, {domain, mode, policy}, &output));
    Check(ref::DecodeLossless(output) == expected, "Pinned full decode mismatch");
    Check(output.size() <= plan.maximum_codestream_bytes &&
              domain->snapshot().peak_backing_bytes <= plan.working.peak_bytes,
          "Transform exceeded storage bound");
    if (!artifacts.empty()) {
      std::ofstream file(artifacts / ("case-" + std::to_string(files) + ".jxl"), std::ios::binary);
      file.write(reinterpret_cast<const char *>(output.data()), output.size());
      Check(bool(file), "Artifact write failed");
    }
    ++files;
  }
  Check(bytes == unchanged, "Transform mutated source");
  ++cases;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc > 1) {
    artifacts = argv[1];
    std::filesystem::create_directories(artifacts);
  }
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    ModularInputProfile profile;
    Ok(ResolveModularInput({1, 1}, format, &profile));
    for (auto extent : {Extent2D{1, 1}, Extent2D{1, 17}, Extent2D{17, 1}, Extent2D{17, 19},
                        Extent2D{257, 3}, Extent2D{3, 257}, Extent2D{2049, 1}})
      for (size_t colors : {size_t{1}, size_t{2}, size_t{16}, size_t{256}}) {
        ModularCodingPolicy policy;
        policy.transforms.size = 1;
        policy.transforms.entries[0] = {TransformKind::kPalette, 0,
                                        static_cast<uint8_t>(profile.channel_count),
                                        static_cast<uint16_t>(colors)};
        Case(extent, format, colors, policy);
      }
  }
  std::cout << "Palette qualified: " << cases << " stages, " << files << " complete files\n"
            << std::flush;
  if (argc > 2 && std::string(argv[2]) == "palette")
    return 0;
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    ModularInputProfile profile;
    Ok(ResolveModularInput({1, 1}, format, &profile));
    for (auto extent :
         {Extent2D{3, 3}, Extent2D{17, 19}, Extent2D{257, 3}, Extent2D{3, 257}, Extent2D{513, 513}})
      for (bool horizontal : {false, true})
        for (bool in_place : {false, true}) {
          ModularCodingPolicy policy;
          policy.rct = profile.channel_count >= 3 ? 6 : 0;
          policy.transforms.size = 1;
          policy.transforms.entries[0] = {TransformKind::kSqueeze,
                                          0,
                                          static_cast<uint8_t>(profile.channel_count),
                                          1,
                                          horizontal,
                                          in_place};
          Case(extent, format, 0, policy);
        }
    ModularCodingPolicy combined;
    combined.rct = profile.channel_count >= 3 ? 9 : 0;
    combined.transforms.size = 3;
    combined.transforms.entries[0] = {TransformKind::kPalette, 0,
                                      static_cast<uint8_t>(profile.channel_count), 16};
    combined.transforms.entries[1] = {TransformKind::kSqueeze, 1, 1, 1, true, false};
    combined.transforms.entries[2] = {TransformKind::kSqueeze, 1, 1, 1, false, true};
    Case({513, 257}, format, 16, combined);
    ModularCodingPolicy dc;
    dc.transforms.size = 6;
    for (size_t i = 0; i < 6; ++i)
      dc.transforms.entries[i] = {TransformKind::kSqueeze,
                                  0,
                                  static_cast<uint8_t>(profile.channel_count),
                                  1,
                                  i % 2 == 0,
                                  false};
    Case({2057, 17}, format, 0, dc);
  }
  for (auto format : {PackedModularFormat::kRgba8, PackedModularFormat::kRgba16}) {
    ModularCodingPolicy p;
    p.transforms.size = 2;
    p.transforms.entries[0] = {TransformKind::kPalette, 0, 3, 16};
    p.transforms.entries[1] = {TransformKind::kPalette, 2, 1, 16};
    Case({257, 3}, format, 16, p);
    p.transforms.size = 1;
    p.transforms.entries[0] = {TransformKind::kPalette, 1, 2, 16};
    Case({3, 257}, format, 16, p);
    p.transforms.size = 2;
    p.transforms.entries[0] = {TransformKind::kSqueeze, 1, 2, 1, true, true};
    p.transforms.entries[1] = {TransformKind::kPalette, 1, 2, 256};
    Case({257, 3}, format, 2, p);
    p.transforms.size = 8;
    for (size_t i = 0; i < 8; ++i)
      p.transforms.entries[i] = {TransformKind::kSqueeze, 0, 4, 1, i % 2 == 0, false};
    Case({2057, 17}, format, 0, p);
  }
  for (bool palette : {false, true}) {
    ModularCodingPolicy p;
    p.tree.size = 3;
    p.tree.nodes[0] = {.property = 0, .split = 1, .left = 1, .right = 2};
    p.tree.nodes[1].predictor = Predictor::kWeighted;
    p.tree.nodes[2].predictor = Predictor::kAverage4;
    p.transforms.size = 1;
    p.transforms.entries[0] = {
        palette ? TransformKind::kPalette : TransformKind::kSqueeze, 0, 4, 256, true, false};
    Case({17, 19}, PackedModularFormat::kRgba16, 16, p);
    Case({257, 3}, PackedModularFormat::kRgba16, 16, p);
  }
  {
    ModularCodingPolicy p;
    p.rct = 6;
    p.transforms.size = 5;
    p.transforms.entries[0] = {TransformKind::kSqueeze, 0, 4, 1, true, false};
    p.transforms.entries[1] = {TransformKind::kSqueeze, 0, 8, 1, false, false};
    p.transforms.entries[2] = {TransformKind::kSqueeze, 0, 16, 1, true, false};
    p.transforms.entries[3] = {TransformKind::kSqueeze, 0, 19, 1, false, false};
    p.transforms.entries[4] = {TransformKind::kSqueeze, 0, 13, 1, true, false};
    Case({33, 35}, PackedModularFormat::kRgba16, 0, p);
    std::array<ChannelShape, kMaximumTransforms + 1> shapes;
    ModularInputProfile profile;
    Ok(ResolveModularInput({33, 35}, PackedModularFormat::kRgba16, &profile));
    Ok(PlanTransforms(profile.channels(), 0, p.transforms, &shapes));
    Check(shapes[5].count == 64, "Channel limit fixture is incomplete");
    p.transforms.size = 6;
    p.transforms.entries[5] = {TransformKind::kSqueeze, 0, 1, 1, true, false};
    Check(!PlanTransforms(profile.channels(), 0, p.transforms, &shapes).ok(),
          "Excess channels accepted");
  }
  std::cout << "Transforms qualified: " << cases << " stages, " << files << " complete files\n";
} catch (const std::exception &e) {
  std::cerr << "case " << cases << ": " << e.what() << '\n';
  return 1;
}
