// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_sample_preparation.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/modular_stream_tokens.h"
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
bool Same(const HistogramTrainingSamples &a,
          const HistogramTrainingSamples &b) {
  return a.total_samples == b.total_samples &&
         a.predictor_count == b.predictor_count &&
         a.property_count == b.property_count &&
         a.static_property_count == b.static_property_count &&
         a.predictors == b.predictors && a.properties == b.properties &&
         a.weights == b.weights && a.residuals == b.residuals &&
         a.values == b.values && a.static_values == b.static_values &&
         a.thresholds == b.thresholds;
}
void Case(GpuBackend &gpu, unsigned kind) {
  auto *op = QueryGpuModularStreamProcessing(gpu);
  auto *generator = dynamic_cast<HistogramRowGenerator *>(&gpu);
  Check(op && generator, "Missing histogram row operation");
  const std::array<Extent2D, 4> shapes =
      kind == 0
          ? std::array<Extent2D, 4>{{{1, 33}, {17, 1}, {33, 41}, {257, 1}}}
          : std::array<Extent2D, 4>{{{3, 5}, {64, 65}, {256, 256}, {7, 9}}};
  const WeightedPredictorParameters wp = {{8, 8, 4, 0, 3, 23, 2},
                                          {13, 12, 12, 12}};
  std::array<std::vector<int32_t>, 4> pixels;
  std::array<ModularChannelView, 4> channels;
  std::array<ModularStreamFeatureSlice, 4> jobs;
  for (size_t c = 0; c < shapes.size(); ++c) {
    const auto [w, h] = shapes[c];
    const size_t stride = w + 3;
    pixels[c].resize(stride * h, 0x12345678);
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        pixels[c][y * stride + x] =
            kind == 2 ? 0
            : kind == 3
                ? (c & 1 ? -1 : 1) * static_cast<int32_t>((y * w + x) * 32768)
                : static_cast<int32_t>(
                      (x * 971 + y * 1831 + x * y * 157 + c * 79) % 65536) -
                      32768;
    channels[c] = {pixels[c], {shapes[c]}, stride};
    if (c == 0)
      channels[c].descriptor.role = ChannelRole::kMetadata;
    // Deliberately distinct from learning-image/channel IDs. The row producer
    // must use the native request's identities, not the feature owner's IDs.
    jobs[c] = {channels[c], wp, static_cast<uint32_t>(600 + c),
               static_cast<uint32_t>(7 - c)};
  }
  ModularStreamFeatureStoragePlan feature_plan;
  Ok(ComputeModularStreamFeatureStoragePlan(jobs, 1, &feature_plan));
  ResourceBudget feature_budget(size_t{64} << 20);
  std::unique_ptr<PreparedGpuModularStreamFeatures> owner;
  {
    ResourceReservation reserve;
    Ok(feature_budget.Reserve(std::max(feature_plan.cpu_preparation.peak_bytes,
                                       feature_plan.gpu_preparation.peak_bytes),
                              &reserve));
    ResourceContextScope scope({&reserve, ResourceClass::kPreparation});
    Ok(op->PrepareModularStreamFeatures(jobs, &owner, 1,
                                        WeightedFeatureBackend::kGpu));
  }
  const std::array<HistogramTrainingImage, 4> images = {
      {{},
       {{256, 256}, channels, 1, wp, owner->weighted_views()},
       {},
       {{256, 256}, channels, 1, wp, owner->weighted_views()}}};
  for (unsigned variant = 0; variant < 4; ++variant) {
    HistogramSamplingOptions options;
    options.maximum_channel_size = 256; // Omit final channel, preserving holes.
    options.maximum_property_values = variant == 0 ? 2 : variant == 1 ? 8 : 256;
    options.sampling_fraction = variant == 0   ? .01f
                                : variant == 3 ? 1.0f
                                               : .65f;
    options.first_group = variant == 3 ? 600 : 7;
    if (variant == 1) {
      options.property_count = 4;
      options.properties = {15, 8, 3, 14};
    }
    if (variant == 2) {
      options.property_count = 4;
      options.properties = {0, 15, 8, 9};
      options.predictor_count = 14;
      for (size_t p = 0; p < 14; ++p)
        options.predictors[p] = static_cast<Predictor>(13 - p);
    }
    PreparedHistogramSamples raster, selected, device;
    Ok(PrepareHistogramSamples(images, &raster, options));
    if (kind == 2 && variant == 2)
      Check(std::ranges::find(raster.samples().weights, UINT16_MAX) !=
                raster.samples().weights.end(),
            "Uniform fixture did not exercise saturated dedup weights");
    options.row_backend = HistogramRowBackend::kSelectedCpu;
    Ok(PrepareHistogramSamples(images, &selected, options));
    Check(Same(raster.samples(), selected.samples()),
          "Selected CPU rows differ from raster oracle");
    options.row_backend = HistogramRowBackend::kGpu;
    options.row_generator = generator;
    options.row_source = owner.get();
    HostStorageBound bound;
    ArmNextManagedHostAllocationFailureForTest();
    Ok(ComputeHistogramSamplePreparationStorageBound(images, options, &bound));
    Check(ManagedHostAllocationFailurePendingForTest(),
          "GPU row storage planning allocated");
    DisarmManagedHostAllocationFailureForTest();
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reserve;
    Ok(budget.Reserve(bound.peak_bytes, &reserve));
    {
      ResourceContextScope scope({&reserve, ResourceClass::kPreparation});
      Ok(PrepareHistogramSamples(images, &device, options));
    }
    Check(Same(raster.samples(), device.samples()),
          "Metal rows, weights or quantizer cuts differ");
    Check(raster.tree_options().split_threshold ==
                  device.tree_options().split_threshold &&
              raster.tree_options().static_range ==
                  device.tree_options().static_range,
          "Metal preparation changed tree options");
    Check(device.stats().rows.gpu_calls == 1 &&
              device.stats().rows.gpu_rows == device.stats().selected_samples &&
              budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
          "Metal row coverage or admitted storage differs");
    // Matching weighted rows alone are insufficient: exercise the fitter too.
    DecisionTree a, b, c;
    Ok(LearnPreparedHistogramTree(raster, &a));
    Ok(LearnPreparedHistogramTree(selected, &b));
    Ok(LearnPreparedHistogramTree(device, &c));
    Check(a == b && a == c, "Sample row route changed the learned tree");
    const auto *old = device.samples().weights.data();
    for (bool completion : {false, true}) {
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, !completion, completion));
      const auto status = PrepareHistogramSamples(images, &device, options);
      Check(!status.ok() && device.samples().weights.data() == old,
            "Failed row submission/completion changed the published samples");
    }
    Ok(ArmNextMetalAllocationFailureForTest(gpu));
    Check(!PrepareHistogramSamples(images, &device, options).ok() &&
              device.samples().weights.data() == old,
          "Failed row allocation changed the published samples");
    device = {};
    reserve.Reset();
    Check(!budget.snapshot().committed_bytes(),
          "Histogram rows leaked managed backing");
  }
  owner.reset();
  Check(!feature_budget.snapshot().committed_bytes(),
        "Histogram rows retained borrowed features");
}
} // namespace
int main() try {
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  for (unsigned kind = 0; kind < 4; ++kind)
    Case(*gpu, kind);
  std::cout << "Selected CPU/Metal rows, dedup weights, cuts, trees, bounds "
               "and failures match\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
