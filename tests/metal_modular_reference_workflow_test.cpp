// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/reference_workflow.h"
#include "codestream/modular/tokenization.h"
#include "gjxl/gjxl.h"
#include "gpu/metal/metal_backend.h"
#if GJXL_REFERENCE_WORKFLOW_ORACLE
#include "modular_reference.h"
#endif
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
using gjxl::codestream_internal::CodestreamBuffer;
void Check(bool b, const char *message) {
  if (!b)
    throw std::runtime_error(message);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
template <typename F> Status Admit(const ResourceBudget &budget, size_t bytes, F &&f) {
  ResourceReservation reservation;
  if (auto s = budget.TryReserve(bytes, &reservation); !s.ok())
    return s;
  ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
  return f();
}
void Released(const ResourceBudget &budget) {
  auto s = budget.snapshot();
  Check(!s.committed_bytes() && !s.total.backing_count && !s.total.pending_count &&
            !s.open_reservations && !s.waiting_requests,
        "Stream workflow leaked resources");
}
DecisionTree Split(unsigned property) {
  DecisionTree tree;
  tree.size = 3;
  tree.nodes[0] = {.property = static_cast<int32_t>(property), .split = 0, .left = 2, .right = 1};
  tree.nodes[1].predictor = Predictor::kAverage4;
  tree.nodes[2].predictor = Predictor::kWeighted;
  tree.nodes[1].offset = -27;
  tree.nodes[2].offset = 19;
  return tree;
}
DecisionTree LargeTree() {
  DecisionTree tree;
  constexpr size_t leaves = 512;
  tree.size = 2 * leaves - 1;
  tree.nodes.resize(tree.size);
  auto fill = [&](auto &&self, size_t i, int32_t lo, int32_t hi) -> void {
    if (lo == hi) {
      tree.nodes[i].predictor = static_cast<Predictor>((i - leaves + 1) % 14);
      return;
    }
    const auto mid = lo + (hi - lo) / 2;
    tree.nodes[i] = {.property = 3,
                     .split = mid,
                     .left = static_cast<TreeIndex>(2 * i + 1),
                     .right = static_cast<TreeIndex>(2 * i + 2)};
    self(self, 2 * i + 1, mid + 1, hi);
    self(self, 2 * i + 2, lo, mid);
  };
  fill(fill, 0, 0, leaves - 1);
  return tree;
}
void Batch(GpuBackend &gpu, bool wavefront) {
  auto *op = QueryGpuModularStreamProcessing(gpu);
  Check(op, "Missing stream generation capability");
  const std::array<Extent2D, 5> shapes = {
      {{17, 13}, {9, 21}, {wavefront ? 127u : 513u, 1}, {1, 33},
       {1, wavefront ? 255u : 257u}}};
  std::array<std::vector<int32_t>, 5> pixels;
  std::array<ModularStreamFeatureSlice, 5> jobs;
  std::array<DecisionTree, 4> tree = {Split(0), Split(1), Split(15), LargeTree()};
  std::array<ModularStreamTokenTree, 5> trees;
  size_t total = 0;
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto [w, h] = shapes[i];
    pixels[i].resize((w + 3) * h, 0x7f7f);
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        pixels[i][y * (w + 3) + x] = int32_t((x * 973 + y * 517 + i * 31) % 65536) - 32768;
    jobs[i] = {{pixels[i], {shapes[i]}, w + 3},
               kSelectionWeightedPresets[i % 3],
               static_cast<uint32_t>(i == 1 ? 65534 : i * 3),
               static_cast<uint32_t>(i % 3)};
    trees[i] = {i, 1, &tree[(i + 2) % tree.size()]};
    total += w * h;
  }
  const auto limits = op->modular_stream_limits();
  ModularStreamFeatureStoragePlan plan;
  ModularStreamTokenStoragePlan token_plan;
  Ok(ComputeModularStreamFeatureStoragePlan(jobs, 1, &plan, &limits));
  Ok(ComputeModularStreamTokenStoragePlan(total, jobs.size(), trees, &token_plan, &limits));
  auto limited = limits;
  limited.maximum_tokens = total - 1;
  ArmNextManagedHostAllocationFailureForTest();
  const auto rejected = ComputeModularStreamFeatureStoragePlan(jobs, 1, &plan, &limited);
  const bool untouched = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(rejected.code() == StatusCode::kUnsupported && untouched && plan.samples == total,
        "Batch preflight allocated or changed the prior plan");
  ResourceBudget budget(size_t{256} << 20);
  std::unique_ptr<PreparedGpuModularStreamFeatures> features;
  std::unique_ptr<PreparedGpuModularTokens> tokens;
  auto prepare = [&] {
    return Admit(budget, std::max(plan.cpu_preparation.peak_bytes, plan.gpu_preparation.peak_bytes), [&] {
      return op->PrepareModularStreamFeatures(jobs, &features, 1, WeightedFeatureBackend::kGpu);
    });
  };
  auto generate = [&] {
    return Admit(budget, token_plan.generation.peak_bytes,
                 [&] { return op->GenerateModularStreamTokens(*features, trees, &tokens); });
  };
  ModularProfile profile;
  const auto submissions = gpu.stats().committed_submissions;
  {
    ProfileSession session(&profile);
    Ok(prepare());
    Ok(generate());
  }
  Check(gpu.stats().committed_submissions == submissions + (wavefront ? 2 : 1) &&
            profile.adaptive_gpu.feature_preparation.gpu_samples == (wavefront ? total : 0) &&
            profile.adaptive_gpu.feature_preparation.cpu_fallback_calls == (wavefront ? 0 : 1),
        "Feature routing or token dispatch count changed");
  const auto host_features = features->weighted_views();
  ModularTokenHostView host_tokens;
  Ok(tokens->BorrowHostView(&host_tokens));
  size_t offset = 0;
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto &job = jobs[i];
    const size_t area = job.input.descriptor.extent.width * job.input.descriptor.extent.height;
    std::vector<WeightedPredictorFeature> expected_features(area);
    std::vector<EntropyToken> expected_tokens(area);
    Ok(PrepareWeightedPredictorFeatures(job.input, job.parameters, expected_features));
    TreeLayout layout;
    Ok(ValidateTree(*trees[i].tree, &layout));
    Ok(TokenizeValidatedModularChannel(job.input, job.channel, job.stream_id,
                                       *trees[i].tree, layout, job.parameters,
                                       expected_tokens));
    Check(std::ranges::equal(expected_features, host_features[i].values),
          "Per-stream weighted parameters or reset boundary changed");
    for (size_t j = 0; j < area; ++j)
      Check(host_tokens.values[offset + j] == expected_tokens[j].value &&
                host_tokens.contexts[offset + j] == expected_tokens[j].context,
            "Batched tree routing, static properties or tokens differ");
    offset += area;
  }
  const auto *old_features = features.get();
  const auto *old_tokens = tokens.get();
  const size_t retained = budget.snapshot().committed_bytes();
  for (bool completion : {false, true}) {
    if (wavefront) {
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, !completion, completion));
      Check(!prepare().ok() && features.get() == old_features &&
                budget.snapshot().committed_bytes() == retained,
            "Feature submission failure published/leaked");
    }
    Ok(ArmNextMetalSubmissionFailureForTest(gpu, !completion, completion));
    Check(!generate().ok() && tokens.get() == old_tokens &&
              budget.snapshot().committed_bytes() == retained,
          "Token submission failure published/leaked");
  }
  Ok(ArmNextMetalAllocationFailureForTest(gpu));
  Check(!prepare().ok() && features.get() == old_features &&
            budget.snapshot().committed_bytes() == retained,
        "Feature allocation failure published/leaked");
  Ok(ArmNextMetalAllocationFailureForTest(gpu));
  Check(!generate().ok() && tokens.get() == old_tokens &&
            budget.snapshot().committed_bytes() == retained,
        "Token allocation failure published/leaked");
  ArmNextManagedHostAllocationFailureForTest();
  const auto failure = prepare();
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(!failure.ok() && !pending && features.get() == old_features &&
            budget.snapshot().committed_bytes() == retained,
        "Feature descriptor allocation failure published/leaked");
  Ok(generate());
  features.reset();
  tokens.reset();
  Released(budget);
}
struct Fixture {
  Extent2D extent;
  PackedModularFormat format;
  SampleByteOrder endian;
  size_t channels, bytes, stride;
  std::vector<uint8_t> packed;
  std::vector<uint16_t> samples;
  Fixture(PackedModularFormat f, Extent2D e, bool palette, SampleByteOrder order)
      : extent(e), format(f), endian(order) {
    ModularInputProfile p;
    Ok(ResolveModularInput(e, f, &p));
    channels = p.channel_count;
    bytes = p.bytes_per_sample;
    stride = e.width * channels * bytes + 3;
    packed.resize(1 + stride * e.height, 0xad);
    uint32_t rng = 3571;
    for (size_t y = 0; y < e.height; ++y)
      for (size_t x = 0; x < e.width; ++x)
        for (size_t c = 0; c < channels; ++c) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          uint32_t v = palette ? ((x / 3 + y / 5 + c) % 5) * (bytes == 1 ? 47 : 12017) : rng;
          v &= bytes == 1 ? 255 : 65535;
          if (c == 3 && (x + y) % 3 == 0)
            v = 0;
          samples.push_back(static_cast<uint16_t>(v));
          const size_t i = 1 + y * stride + (x * channels + c) * bytes;
          packed[i] = bytes == 2 && endian == SampleByteOrder::kBigEndian ? v >> 8 : v;
          if (bytes == 2)
            packed[i + 1] = endian == SampleByteOrder::kBigEndian ? v : v >> 8;
        }
  }
  PackedModularImageView view() const {
    return {std::span(packed).subspan(1), extent, stride, format, endian};
  }
};
size_t cases = 0, samples_checked = 0;
void WorkflowFailures(GpuBackend &gpu) {
  Fixture f(PackedModularFormat::kGray8, {17, 19}, true, SampleByteOrder::kLittleEndian);
  ResourceBudget budget(size_t{256} << 20);
  ReferenceModularFrame prior;
  Ok(PrepareReferenceModularFrame(f.view(), {}, budget, &prior));
  const auto *old = prior.streams().data();
  const auto retained = budget.snapshot().committed_bytes();
  ReferenceWorkflowOptions options;
  options.gpu = &gpu;
  options.feature_producer = WeightedFeatureBackend::kGpu;
  for (bool completion : {false, true}) {
    Ok(ArmNextMetalSubmissionFailureForTest(gpu, !completion, completion));
    Check(!PrepareReferenceModularFrame(f.view(), options, budget, &prior).ok() &&
              prior.streams().data() == old && budget.snapshot().committed_bytes() == retained,
          "Workflow device failure changed prior frame or leaked features");
  }
  options.maximum_samples = 1;
  Check(PrepareReferenceModularFrame(f.view(), options, budget, &prior).code() ==
                StatusCode::kUnsupported &&
            prior.streams().data() == old && budget.snapshot().committed_bytes() == retained,
        "Learning rejection did not release completed features");
  options.maximum_samples = kHistogramMaximumSamples;
  ResourceBudget tiny(1), busy(size_t{256} << 20);
  ResourceReservation hold;
  Ok(busy.TryReserve(size_t{256} << 20, &hold));
  Check(PrepareReferenceModularFrame(f.view(), options, tiny, &prior).code() ==
                StatusCode::kOutOfMemory &&
            PrepareReferenceModularFrame(f.view(), options, busy, &prior).code() ==
                StatusCode::kUnavailable &&
            prior.streams().data() == old,
        "Metal workflow admission changed prior output");
  prior = {};
  Released(budget);
  Released(tiny);
}
void Workflow(GpuBackend &gpu, const Fixture &f, ReferenceWorkflowOptions options = {}) {
  ResourceBudget budget(size_t{2} << 30);
  ReferenceModularFrame cpu, metal;
  auto cpu_options = options;
  cpu_options.fitting_backend = ReferenceFittingBackend::kCpu;
  Ok(PrepareReferenceModularFrame(f.view(), cpu_options, budget, &cpu));
  options.gpu = &gpu;
  if (f.extent == Extent2D{257, 257})
    options.feature_producer = WeightedFeatureBackend::kGpu;
  ModularProfile profile;
  {
    ProfileSession session(&profile);
    Ok(PrepareReferenceModularFrame(f.view(), options, budget, &metal));
  }
  Check(cpu.stats().learning_calls == metal.stats().learning_calls &&
            cpu.stats().fits == metal.stats().fits &&
            cpu.stats().scanned_pixels == metal.stats().scanned_pixels &&
            metal.stats().scanned_pixels == metal.stats().feature_samples &&
            metal.stats().feature_samples == metal.stats().token_samples &&
            profile.training_work.weighted_prediction_samples == 0 &&
            profile.training_work.reused_prediction_samples == metal.stats().scanned_pixels &&
            profile.adaptive_gpu.feature_preparations == metal.stats().feature_batches &&
            profile.adaptive_gpu.feature_reuses == metal.stats().token_batches,
        "Learned Metal refitted or regenerated weighted features");
  if (options.fitting_backend == ReferenceFittingBackend::kGpu &&
      f.extent == Extent2D{769, 513})
    Check(metal.stats().local_trees && metal.stats().histogram_gpu_nodes > 0,
          "Local workflow did not exercise GPU histogram fitting");
  if (options.fitting_backend == ReferenceFittingBackend::kAuto)
    Check(!metal.stats().histogram_gpu_nodes && !metal.stats().histogram_cpu_nodes,
          "Small shared fit did not retain the ordinary CPU fitter");
  if (f.extent == Extent2D{257, 257})
    Check(profile.adaptive_gpu.feature_preparation.gpu_calls == 1 &&
              profile.adaptive_gpu.feature_preparation.gpu_samples > 0,
          "Forced feature producer did not dispatch the full-size slices");
  for (size_t i = 0; i < cpu.streams().size(); ++i) {
    Check(cpu.streams()[i].tree == metal.streams()[i].tree &&
              cpu.streams()[i].selection.wp_mode == metal.streams()[i].selection.wp_mode &&
              metal.view().streams[i].weighted_features.empty(),
          "Learned tree/WP differs or stale borrow retained");
  }
  PreparedModularStreamTokens expected;
  Ok(TokenizePreparedModularStreams(cpu.view(), &expected));
  PreparedModularStreamTokens actual;
  Ok(TokenizePreparedModularStreams(metal.view(), &actual));
  const auto *values = actual.streams[0].values.data();
  ReferenceModularFrame moved(std::move(metal));
  metal = std::move(moved);
  PreparedModularStreamTokens after_move;
  Ok(TokenizePreparedModularStreams(metal.view(), &after_move));
  Check(values == after_move.streams[0].values.data(),
        "Frame move invalidated token backing");
  for (size_t i = 0; i < expected.streams.size(); ++i) {
    const auto stream = actual.streams[i];
    Check(expected.streams[i].size() == stream.size(), "Canonical stream coverage differs");
    for (size_t j = 0; j < stream.size(); ++j)
      Check(expected.streams[i][j] == stream[j], "Learned GPU token differs");
  }
  for (bool prefix : {false, true}) {
    auto entropy = options.entropy;
    if (prefix)
      entropy = {PreparedModularEntropyPolicy::kPrefix};
    CodestreamBuffer a, b;
    for (bool on_gpu : {false, true}) {
      auto view = on_gpu ? metal.view() : cpu.view();
      ModularWorkflowStoragePlan plan;
      Ok(ComputePreparedModularStreamsStoragePlan(view, entropy, &plan));
      Ok(Admit(budget, plan.working.peak_bytes,
               [&] { return EncodePreparedModularStreams(view, entropy, on_gpu ? &b : &a); }));
    }
    Check(std::ranges::equal(a.view(), b.view()), "Learned Metal codestream bytes differ");
#if GJXL_REFERENCE_WORKFLOW_ORACLE
    const test::modular_reference::IntegerImage original{
        f.extent.width, f.extent.height, static_cast<uint32_t>(f.channels),
        static_cast<uint32_t>(f.bytes * 8), f.samples};
    Check(test::modular_reference::DecodeLossless(b.view()) == original,
          "Learned Metal exact decode differs");
#endif
    ++cases;
  }
  samples_checked += metal.stats().token_samples;
  metal = {};
  cpu = {};
  Released(budget);
}
void PublicC() {
  GJXLContextOptions context_options;
  Check(gjxl_context_options_init(&context_options, sizeof(context_options)) == GJXL_OK,
        "C context options failed");
  std::array<uint8_t, 9> pixels = {1, 7, 4, 3, 5, 2, 9, 6, 8};
  GJXLImageView view{sizeof(GJXLImageView), 3, 3, GJXL_PIXEL_FORMAT_GRAY8_SRGB, pixels.data(),
                     pixels.size(),         3};
  std::vector<uint8_t> expected;
  for (auto backend : {GJXL_BACKEND_CPU, GJXL_BACKEND_METAL}) {
    context_options.backend = backend;
    GJXLContext *context = nullptr;
    Check(gjxl_context_create(&context_options, &context) == GJXL_OK, gjxl_get_last_error());
    GJXLBuffer output{};
    Check(gjxl_encode_modular(context, &view, nullptr, &output) == GJXL_OK, gjxl_get_last_error());
    if (backend == GJXL_BACKEND_CPU)
      expected.assign(output.data, output.data + output.size);
    else
      Check(std::ranges::equal(expected, std::span(output.data, output.size)),
            "C Metal bytes differ");
    gjxl_buffer_free(&output);
    gjxl_context_destroy(context);
  }
}
} // namespace
int main() try {
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  Batch(*gpu, false);
  Batch(*gpu, true);
  WorkflowFailures(*gpu);
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16})
    for (bool palette : {false, true})
      Workflow(*gpu,
               Fixture(format, {17, 19}, palette,
                       palette ? SampleByteOrder::kLittleEndian : SampleByteOrder::kBigEndian));
  for (Extent2D extent : {Extent2D{1, 33}, Extent2D{257, 3}, Extent2D{1793, 3}, Extent2D{2049, 3},
                          Extent2D{257, 257}})
    Workflow(*gpu,
             Fixture(PackedModularFormat::kRgba16, extent, true, SampleByteOrder::kBigEndian));
  ReferenceWorkflowOptions buffered;
  buffered.buffering = ReferenceBuffering::kBuffered;
  buffered.predictors = ReferencePredictors::kAll;
  Workflow(*gpu,
           Fixture(PackedModularFormat::kRgb8, {2049, 3}, false, SampleByteOrder::kLittleEndian),
           buffered);
  ReferenceWorkflowOptions histogram;
  histogram.fitting_backend = ReferenceFittingBackend::kGpu;
  histogram.fitting_minimum_rows = 1;
  histogram.fitting_batch_fits = 3;
  for (auto format : {PackedModularFormat::kRgb8, PackedModularFormat::kRgba16})
    for (bool palette : {false, true})
      Workflow(*gpu, Fixture(format, {17, 19}, palette, SampleByteOrder::kBigEndian), histogram);
  histogram.fitting_minimum_rows = 1024;
  Workflow(*gpu, Fixture(PackedModularFormat::kRgb8, {769, 513}, false,
                         SampleByteOrder::kLittleEndian), histogram);
  histogram.fitting_minimum_rows = 1;
  histogram.predictors = ReferencePredictors::kAll;
  Workflow(*gpu, Fixture(PackedModularFormat::kRgba16, {2049, 3}, false,
                         SampleByteOrder::kBigEndian), histogram);
  histogram.buffering = ReferenceBuffering::kBuffered;
  Workflow(*gpu, Fixture(PackedModularFormat::kRgb16, {257, 3}, false,
                         SampleByteOrder::kLittleEndian), histogram);
  histogram.fitting_backend = ReferenceFittingBackend::kAuto;
  Workflow(*gpu, Fixture(PackedModularFormat::kRgb8, {17, 19}, false,
                         SampleByteOrder::kLittleEndian), histogram);
  PublicC();
  std::cout << cases << " CPU/Metal codestream pairs and " << samples_checked
            << " learned tokens matched; mixed WP/tree batches and failure cleanup passed\n";
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n';
  return 1;
}
