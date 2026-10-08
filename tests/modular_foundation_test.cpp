// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "codec/modular/image.h"
#include "codestream/modular/stream_plan.h"
#include "codestream/headers_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}

void ViewsAndImages() {
  std::vector<int32_t> backing(60, 65535);
  ModularChannelView view{std::span(backing).subspan(3), {{7, 5}}, 11};
  Ok(view.Validate());
  ModularChannelView slice;
  Ok(BorrowChannelSlice(view, {2, 1, {4, 3}}, &slice));
  Check(slice.Row(2) == view.Row(3) + 2 && slice.Row(2)[3] == 65535,
        "slice copied or misaddressed");
  const auto *sentinel = slice.backing.data();
  Check(!BorrowChannelSlice(view, {6, 0, {2, 1}}, &slice).ok() &&
            slice.backing.data() == sentinel,
        "invalid slice mutated output");
  view.backing = view.backing.first(50);
  Check(!view.Validate().ok(), "short backing accepted");
  view.stride = SIZE_MAX;
  Check(!view.Validate().ok(), "overflowing stride accepted");

  const std::array channels = {ChannelDescriptor{{2, 1}, 0, 0, ChannelRole::kMetadata},
                               ChannelDescriptor{{7, 5}},
                               ChannelDescriptor{{4, 3}, 1, 1}};
  HostStorageBound bound;
  Ok(ComputeModularImageStorageBound(channels, 1, &bound));
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeModularImageStorageBound(channels, 1, &bound));
  Check(ManagedHostAllocationFailurePendingForTest(),
        "image planner allocated backing");
  DisarmManagedHostAllocationFailureForTest();
  auto prior = bound;
  Check(!ComputeModularImageStorageBound(channels, 0, &bound).ok() && bound == prior,
        "invalid metadata prefix accepted");
  const std::array overflow = {ChannelDescriptor{{SIZE_MAX, 2}}};
  Check(!ComputeModularImageStorageBound(overflow, 0, &bound).ok() && bound == prior,
        "overflowing image plan changed output");
  // Old output overlaps every replacement candidate; both are admitted.
  ResourceBudget budget(bound.peak_bytes * 2);
  ResourceReservation reservation;
  Ok(budget.Reserve(bound.peak_bytes * 2, &reservation));
  {
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    ModularImage image;
    Ok(ModularImage::Create(channels, 1, &image));
    image.samples(1)[0] = 123;
    for (size_t fail = 0; fail < 4; ++fail) {
      ArmManagedHostAllocationFailureAfterForTest(fail);
      const auto s = ModularImage::Create(channels, 1, &image);
      DisarmManagedHostAllocationFailureForTest();
      Check(s.code() == StatusCode::kOutOfMemory && image.samples(1)[0] == 123,
            "partial image replacement was published");
      Check(budget.snapshot().total.live_capacity_bytes == bound.retained_bytes,
            "partial image allocation leaked");
    }
    Ok(ModularImage::Create(channels, 1, &image));
    Check(image.samples(1)[0] == 0 && image.metadata_channels() == 1,
          "image initialization is wrong");
    Check(budget.snapshot().peak_backing_bytes <= 2 * bound.peak_bytes,
          "image exceeded replacement bound");
    Check(budget.snapshot()
                  .classes[static_cast<size_t>(ResourceClass::kPreparation)]
                  .live_capacity_bytes == bound.retained_bytes,
          "image lost allocation owner");
  }
  Check(budget.snapshot().total.backing_count == 0, "image storage leaked");
}

void GeometryAndPlans() {
  for (Extent2D extent : {Extent2D{1, 1},
                          {1, 257},
                          {257, 1},
                          {17, 19},
                          {255, 256},
                          {256, 256},
                          {257, 257},
                          {2047, 1},
                          {2048, 1},
                          {2049, 2049}}) {
    ModularFrameGeometry g;
    Ok(ModularFrameGeometry::Create(extent, &g));
    const auto expected_groups = extent.ceil_div(256);
    Check(g.groups() == expected_groups, "wrong group grid");
    const std::array channels = {ChannelDescriptor{extent}, ChannelDescriptor{extent}};
    ModularStreamStoragePlan bound;
    Ok(ComputeModularStreamStoragePlan(g, channels, 0, &bound));
    ResourceBudget budget(bound.owned.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.owned.peak_bytes, &reservation));
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      ModularStreamPlan p;
      for (size_t fail = 0; fail < 2; ++fail) {
        ArmManagedHostAllocationFailureAfterForTest(fail);
        auto s = BuildModularStreamPlan(g, channels, 0, &p);
        DisarmManagedHostAllocationFailureForTest();
        Check(s.code() == StatusCode::kOutOfMemory && p.streams.empty() &&
                  p.slices.empty(),
              "partial stream plan escaped");
      }
      Ok(BuildModularStreamPlan(g, channels, 0, &p));
      size_t area = 0;
      for (const auto &slice : p.slices)
        area += slice.rect.extent.width * slice.rect.extent.height;
      Check(area == 2 * extent.width * extent.height,
            "channel samples lost or duplicated");
      Check(p.streams[0].slice_count == (g.group_count() == 1 ? 2 : 0),
            "global placement wrong");
      for (const auto &stream : p.streams) {
        Check(stream.section < g.section_count(), "section index outside TOC");
        if (stream.role == StreamRole::kDcGroup)
          Check(stream.slice_count == 0, "identity DC stream not empty");
      }
      Check(budget.snapshot().peak_backing_bytes <= bound.owned.peak_bytes,
            "plan storage overrun");
    }
    Check(budget.snapshot().total.backing_count == 0, "plan leaked");
    const auto before = g;
    Check(!ModularFrameGeometry::Create({0, 1}, &g).ok() && g == before &&
              !ModularFrameGeometry::Create({SIZE_MAX, 1}, &g).ok() && g == before,
          "invalid geometry changed output");
  }
  ModularFrameGeometry g;
  Ok(ModularFrameGeometry::Create({4097, 4097}, &g));
  const std::array shifted = {
      ChannelDescriptor{{3, 1}, 0, 0, ChannelRole::kMetadata},
      ChannelDescriptor{{129, 129}, 5, 5}, ChannelDescriptor{{513, 513}, 3, 3},
      ChannelDescriptor{{2049, 2048}, 1, 1}, ChannelDescriptor{{1, 1}}};
  ModularStreamPlan p;
  Ok(BuildModularStreamPlan(g, shifted, 1, &p));
  Check(p.streams[0].slice_count == 2, "global prefix included later small channel");
  std::array<size_t, 5> areas{};
  for (const auto &stream : p.streams)
    for (size_t i = 0; i < stream.slice_count; ++i) {
      const auto &slice = p.slices[stream.slice_begin + i];
      if (slice.channel == 2)
        Check(stream.role == StreamRole::kDcGroup, "shifted DC placement wrong");
      if (slice.channel >= 3)
        Check(stream.role == StreamRole::kGroup, "shifted group placement wrong");
      areas[slice.channel] += slice.rect.extent.width * slice.rect.extent.height;
    }
  for (size_t c = 0; c < shifted.size(); ++c)
    Check(areas[c] == shifted[c].extent.width * shifted[c].extent.height,
          "shifted coverage wrong");
  size_t id = 99;
  Check(!ModularStreamId(g, static_cast<StreamRole>(255), 0, &id).ok() && id == 99,
        "unknown stream kind accepted");
  const auto previous_streams = std::vector(p.streams.begin(), p.streams.end());
  for (size_t fail = 0; fail < 2; ++fail) {
    ArmManagedHostAllocationFailureAfterForTest(fail);
    const auto status = BuildModularStreamPlan(g, shifted, 1, &p);
    DisarmManagedHostAllocationFailureForTest();
    Check(status.code() == StatusCode::kOutOfMemory &&
              std::ranges::equal(p.streams, previous_streams),
          "failed replacement lost previous stream plan");
  }
  ModularStreamStoragePlan bound;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeModularStreamStoragePlan(g, shifted, 1, &bound));
  Check(ManagedHostAllocationFailurePendingForTest(),
        "stream planner allocated backing");
  DisarmManagedHostAllocationFailureForTest();
  const auto old_bound = bound;
  const std::array invalid = {ChannelDescriptor{{4098, 4097}}};
  Check(!ComputeModularStreamStoragePlan(g, invalid, 0, &bound).ok() &&
            bound == old_bound,
        "invalid stream bound changed output");
  const std::array excessive_shift = {ChannelDescriptor{{4097, 1}, 0, 9}};
  Check(!ComputeModularStreamStoragePlan(g, excessive_shift, 0, &bound).ok(),
        "unrepresentable grouped channel shift accepted");
}

void HeadersAndBounds() {
  using namespace codec_internal;
  using namespace codestream_internal;
  for (uint8_t bits : {8, 16})
    for (SourceColor color : {SourceColor::kSrgb, SourceColor::kGraySrgb})
      for (bool alpha : {false, true}) {
        if (alpha && color == SourceColor::kGraySrgb)
          continue;
        ImageMetadata m{{257, 2049}, SampleFormat::kUnsigned, bits, color};
        if (alpha)
          m.alpha = AlphaMetadata{bits};
        CommonHeaderStoragePlan plan;
        Ok(ComputeCommonHeaderStoragePlan(7, &plan));
        auto bound = plan.destination;
        Check(bound.Add(plan.temporary), "header bound overflow");
        ResourceBudget budget(bound.peak_bytes);
        ResourceReservation reservation;
        Ok(budget.Reserve(bound.peak_bytes, &reservation));
        {
          ResourceContextScope scope({&reservation, ResourceClass::kSerializer});
          for (size_t offset = 0; offset < 8; ++offset) {
            BitWriter w;
            Ok(w.WriteBits(offset, 0));
            Ok(WriteImageHeader(m, &w));
            Check(w.bits_written() - offset <= plan.maximum_image_bits,
                  "image header overrun");
            const size_t start = w.bits_written();
            Ok(WriteFrameHeader(m, {}, &w));
            Check(w.bits_written() - start <= plan.maximum_frame_bits,
                  "frame header overrun");
          }
          Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
                "header storage overrun");
        }
        for (bool frame_header : {false, true})
          for (size_t fail = 0; fail < 8; ++fail) {
            BitWriter w;
            Ok(w.WriteBits(3, 5));
            ArmManagedHostAllocationFailureAfterForTest(fail);
            auto s =
                frame_header ? WriteFrameHeader(m, {}, &w) : WriteImageHeader(m, &w);
            DisarmManagedHostAllocationFailureForTest();
            if (!s.ok())
              Check(w.bits_written() == 3 && w.padded_bytes()[0] == 5,
                    "header failure changed writer");
          }
        m.bits = 7;
        BitWriter w;
        Ok(w.WriteBits(3, 5));
        Check(!WriteImageHeader(m, &w).ok() && !WriteFrameHeader(m, {}, &w).ok() &&
                  w.bits_written() == 3 && w.padded_bytes()[0] == 5,
              "unsupported header changed writer");
      }
  const ImageMetadata supported{{1, 1}};
  for (auto invalid :
       {ImageMetadata{{0, 1}}, ImageMetadata{{1, 1}, static_cast<SampleFormat>(255)},
        ImageMetadata{
            {1, 1}, SampleFormat::kUnsigned, 8, static_cast<SourceColor>(255)},
        ImageMetadata{{1, 1}, SampleFormat::kUnsigned, 8, SourceColor::kSrgb, true},
        ImageMetadata{{1, 1},
                      SampleFormat::kUnsigned,
                      8,
                      SourceColor::kSrgb,
                      false,
                      AlphaMetadata{8, true}},
        ImageMetadata{{1, 1},
                      SampleFormat::kUnsigned,
                      8,
                      SourceColor::kSrgb,
                      false,
                      AlphaMetadata{16}}})
    Check(!ValidateImageMetadata(invalid).ok(), "unsupported image metadata accepted");
  Check(!ValidateFrameMetadata(supported, {static_cast<FrameEncoding>(255)}).ok() &&
            !ValidateFrameMetadata(supported, {FrameEncoding::kVarDct}).ok() &&
            !ValidateFrameMetadata(supported, {FrameEncoding::kModular, {3, 2}}).ok(),
        "invalid frame mode accepted");
  CommonHeaderStoragePlan bound;
  const auto unchanged = bound;
  Check(!ComputeCommonHeaderStoragePlan(SIZE_MAX, &bound).ok() && bound == unchanged,
        "header overflow changed plan");
  BitWriter limited;
  Ok(limited.WriteBits(3, 5));
  Check(!limited.WithMaxBits(0, [&] { return WriteImageHeader(supported, &limited); })
                .ok() &&
            limited.bits_written() == 3 && limited.padded_bytes()[0] == 5,
        "header escaped parent allotment");
}
} // namespace
int main() {
  try {
    ViewsAndImages();
    GeometryAndPlans();
    HeadersAndBounds();
    std::cout << "Modular channel, geometry, storage and header checks passed\n";
  } catch (const std::exception &e) {
    DisarmManagedHostAllocationFailureForTest();
    std::cerr << e.what() << '\n';
    return 1;
  }
}
