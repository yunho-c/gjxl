// SPDX-License-Identifier: Apache-2.0
// A standalone capture compiled against matching baseline/candidate sources.
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "codestream_frame_fixture.h"
#include "sparse_frame_fixture.h"
#include "codec/dc_quantization.h"
#include "codec/color_transform.h"
#include "codec/quantization_pipeline.h"
#include "codestream/dc_context_tree_internal.h"
#include "codestream/dc_group.h"
#include "codestream/encoder_internal.h"
#include "codestream/serializer_storage_plan.h"
#include "codestream/workflow.h"
#include "core/image_buffer.h"
#include "core/thread_budget.h"
#include "io/pfm.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using gjxl_test::Check;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
std::string case_filter;

bool Selected(const std::string& name) {
  return case_filter.empty() ||
      ("," + case_filter + ",").find("," + name + ",") != std::string::npos;
}

uint64_t Nanoseconds(Clock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
}

struct Snapshot {
  std::ofstream file;
  explicit Snapshot(const fs::path& path) : file(path, std::ios::binary) {
    file.exceptions(std::ios::badbit | std::ios::failbit);
  }
  void Number(uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) file.put(char(value >> shift));
  }
  void Bound(const resource_budget_internal::HostStorageBound& bound) {
    Number(bound.retained_bytes);
    Number(bound.peak_bytes);
  }
  void Tokens(std::span<const EntropyToken> tokens) {
    Number(tokens.size());
    for (auto token : tokens) { Number(token.context); Number(token.value); }
  }
};

void Bytes(const fs::path& path, std::span<const uint8_t> bytes) {
  std::ofstream file(path, std::ios::binary);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void RecordFrame(const fs::path& output, const std::string& name,
                 vardct_frame_internal::VarDctFrameView frame,
                 VarDctCodestreamOptions options, size_t repeats,
                 std::ofstream& timings) {
  if (!Selected(name)) return;
  thread_budget_internal::EncodeScope serial(1);
  Snapshot record(output / (name + ".snapshot"));
  record.Number(frame.geometry().frame().width);
  record.Number(frame.geometry().frame().height);
  size_t samples = 0;
  Check(ComputeDcSampleCount(frame.geometry().block_grid().blocks, &samples));
  const DcContextTreeLayout* layout = nullptr;
  Check(SelectDcContextTreeLayout(samples, options.dc_prediction,
                                 CurrentDcTreePolicy(), &layout));
  record.Number(layout->dc_leaf_count);
  record.Number(layout->context_count);
  record.Tokens(layout->tree_tokens());
  for (auto value : layout->context_map) record.Number(value);
  Storage<SimpleDcGroupTokenStreams> groups;
  Check(TokenizeSimpleDcGroupsForEncoder(frame, &groups, options.dc_prediction));
  record.Number(groups.size());
  for (const auto& group : groups) {
    record.Number(group.block_x); record.Number(group.block_y);
    record.Number(group.block_extent.width); record.Number(group.block_extent.height);
    record.Number(group.transform_anchor_count);
    record.Tokens(group.dc_tokens); record.Tokens(group.ac_metadata_tokens);
  }
  SerializerStoragePlan plan;
  Check(ComputeSerializerStoragePlan(frame.geometry().frame(),
      {.coding = options, .cpu_thread_count = 1, .collect_profile = true}, &plan));
  for (auto value : {plan.ac_group_count, plan.dc_group_count, plan.maximum_order_variants,
                    plan.maximum_ac_candidates, plan.maximum_ac_tokens,
                    plan.maximum_dc_tokens, plan.maximum_output_bytes}) record.Number(value);
  record.Bound(plan.output); record.Bound(plan.working);
  TokenizationStoragePlan tokens;
  Check(ComputeTokenizationStoragePlan(frame.geometry().block_grid().blocks,
      {.context_count = 1485, .workers = 1, .dc_prediction = options.dc_prediction}, &tokens));
  record.Number(tokens.maximum_ac_tokens); record.Number(tokens.maximum_dc_tokens);
  record.Bound(tokens.dc); record.Bound(tokens.ac);

  std::vector<uint8_t> reference;
  VarDctCodestreamProfile profile;
  Check(EncodeVarDctCodestreamFromView(frame, options, &reference, &profile));
  for (uint64_t value : {uint64_t(profile.dc_leaf_count), uint64_t(profile.dc_context_count),
       profile.entropy_model_bits, profile.entropy_token_bits,
       uint64_t(profile.dc_entropy_clusters), uint64_t(profile.ac_entropy_clusters),
       uint64_t(profile.dc_entropy_is_ans), uint64_t(profile.ac_entropy_is_ans),
       uint64_t(profile.coefficient_order_entropy_is_ans),
       uint64_t(profile.natural_candidate_bytes), uint64_t(profile.custom_order_candidate_bytes),
       uint64_t(profile.balanced_candidate_bytes), uint64_t(profile.rate_candidate_bytes),
       uint64_t(profile.selected_balanced_fallback), uint64_t(profile.selected_coefficient_order_mask),
       uint64_t(profile.selected_block_context_candidate_index),
       uint64_t(profile.selected_block_context_count)}) record.Number(value);
  Bytes(output / (name + ".jxl"), reference);
  for (size_t round = 0; round < repeats; ++round) {
    auto begin = Clock::now();
    Storage<SimpleDcGroupTokenStreams> current;
    Check(TokenizeSimpleDcGroupsForEncoder(frame, &current, options.dc_prediction));
    auto dc_time = Nanoseconds(begin);
    if (current != groups) throw std::runtime_error("DC capture is not repeatable");
    std::vector<uint8_t> encoded;
    begin = Clock::now();
    Check(EncodeVarDctCodestreamFromView(frame, options, &encoded));
    auto encode_time = Nanoseconds(begin);
    if (encoded != reference) throw std::runtime_error("Profiled/ordinary bytes differ");
    timings << name << ',' << round << ",dc," << dc_time << '\n';
    timings << name << ',' << round << ",serializer," << encode_time << '\n';
  }
}

void Synthetic(const fs::path& output, size_t repeats, std::ofstream& timings) {
  constexpr std::array extents = {Extent2D{4, 4}, Extent2D{16, 16}, Extent2D{40, 40},
      Extent2D{64, 64}, Extent2D{260, 4}, Extent2D{260, 260}};
  size_t index = 0;
  for (auto blocks : extents) {
    codestream_test_internal::FrameFixture fixture;
    if (!fixture.Create(blocks, 7, index == 2 ? 1 : 0))
      throw std::runtime_error("Cannot create capture frame");
    for (size_t i = 0; i < fixture.zero_dc.size(); ++i)
      fixture.zero_dc[i] = static_cast<int32_t>((i * 73 + i / blocks.width * 19) % 1021) - 510;
    std::array<std::vector<float>, 3> reconstruction;
    ConstImage3FView dc;
    const auto steps = fixture.quantizer.dc_steps();
    for (size_t c = 0; c < 3; ++c) {
      reconstruction[c].resize(fixture.zero_dc.size());
      for (size_t i = 0; i < fixture.zero_dc.size(); ++i) {
        reconstruction[c][i] = static_cast<float>(fixture.zero_dc[i]) * steps[c];
        if (c == 2) reconstruction[c][i] += static_cast<float>(fixture.zero_dc[i]) * steps[1];
      }
      dc.plane[c] = {reconstruction[c].data(), blocks, blocks.width};
    }
    vardct_frame_internal::VarDctFrameView frame({
        .input = {.geometry = fixture.geometry, .strategies = &fixture.strategies,
                  .raw_quant_field = {fixture.quant.data(), blocks, blocks.width},
                  .quantizer = &fixture.quantizer, .color_correlation = &fixture.cfl,
                  .epf_sharpness = {fixture.sharpness.data(), blocks, blocks.width}},
        .quantized_dc = fixture.view().quantized_dc(), .dc = dc,
        .ac_group_extent = fixture.groups, .group_used_coefficient_count = fixture.used,
        .ac_coefficients = fixture.coefficients});
    for (auto prediction : {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted}) {
      for (auto policy : {DcTreePolicy::kAdaptive, DcTreePolicy::kLegacy}) {
        ScopedDcTreePolicyForTesting select(policy);
        const auto name = "frame_" + std::to_string(index) + "_p" +
            std::to_string(int(prediction)) + "_t" + std::to_string(int(policy));
        RecordFrame(output, name, frame, {.dc_prediction = prediction}, repeats, timings);
      }
      if (index < 2) {
        for (auto entropy : {VarDctEntropyBehavior::kHighDensity,
                             VarDctEntropyBehavior::kMaximumCompression,
                             VarDctEntropyBehavior::kRateOptimized}) {
          const auto name = "entropy_" + std::to_string(index) + "_p" +
              std::to_string(int(prediction)) + "_e" + std::to_string(int(entropy));
          RecordFrame(output, name, frame, {.entropy_behavior = entropy,
              .coefficient_order_behavior = VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled,
              .dc_prediction = prediction, .dc_uint_search = true}, repeats, timings);
        }
      }
    }
    ++index;
  }
}

template <typename T>
void Representations(const fs::path& output, size_t repeats, std::ofstream& timings) {
  auto seed = gjxl_test::MakeFrame(7, 2, 36);
  gjxl_test::PopulationAssembly assembly(seed);
  std::vector<T> coefficients(assembly.coefficients.begin(), assembly.coefficients.end());
  vardct_frame_internal::QuantizedFrameAssemblyInputT<T> input{
      .geometry = seed.geometry(), .strategies = &seed.strategies(),
      .raw_quant_field = seed.raw_quant_field(), .quantizer = &seed.quantizer(),
      .y_to_x = seed.color_correlation().y_to_x_map(),
      .y_to_b = seed.color_correlation().y_to_b_map(), .epf_sharpness = seed.epf_sharpness(),
      .profile = seed.profile(), .quantized_dc = seed.quantized_dc(),
      .quantized_ac = coefficients, .transforms = assembly.layouts};
  VarDctEncoderFrame compact, sparse;
  Check(vardct_frame_internal::AssembleVarDctEncoderFrame(input, &compact));
  auto sparse_storage = gjxl_test::SparseStorage<T>(seed);
  Check(vardct_frame_internal::AssembleVarDctEncoderFrame(
      gjxl_test::SparseInput(seed, assembly.layouts, &sparse_storage), &sparse));
  RecordFrame(output, "compact_" + std::to_string(sizeof(T)),
              vardct_frame_internal::BorrowFrame(compact), {}, repeats, timings);
  RecordFrame(output, "sparse_" + std::to_string(sizeof(T)),
              vardct_frame_internal::BorrowFrame(sparse), {}, repeats, timings);
}

void Workflow(const fs::path& output, const std::string& name, ConstImage3FView image,
              VarDctEncodingOptions options, size_t repeats, std::ofstream& timings) {
  if (!Selected(name)) return;
  options.backend = VarDctBackendPreference::kCpu;
  options.cpu_thread_count = 1;
  std::vector<uint8_t> reference;
  VarDctEncodingSummary summary;
  Check(EncodeLinearRgbVarDctCodestream(image, options, &reference, &summary));
  Bytes(output / (name + ".jxl"), reference);
  Snapshot record(output / (name + ".snapshot"));
  record.Number(summary.encoded_bytes);
  record.Number(int(summary.entropy_behavior));
  record.Number(int(summary.dc_prediction)); record.Number(int(summary.dc_quantization));
  record.Number(summary.adaptive_dc_smoothing);
  record.Number(summary.encode_attempt_count); record.Number(summary.failed_encode_attempt_count);
  record.Number(std::bit_cast<uint32_t>(summary.selected_butteraugli_target));
  record.Number(summary.target_size_met); record.Number(summary.score_history.size());
  for (auto count : summary.strategy_counts) record.Number(count);
  for (auto score : summary.score_history) record.Number(std::bit_cast<uint64_t>(score));
  for (size_t round = 0; round < repeats; ++round) {
    std::vector<uint8_t> bytes;
    VarDctEncodingSummary actual;
    auto begin = Clock::now();
    Check(EncodeLinearRgbVarDctCodestream(image, options, &bytes, &actual));
    auto elapsed = Nanoseconds(begin);
    if (bytes != reference || actual != summary) throw std::runtime_error("Workflow not repeatable");
    timings << name << ',' << round << ",workflow," << elapsed << '\n';
  }
}
// Opt-in photographic stage capture. Preparation is outside the measurements;
// both predictors serialize the same owned CPU frame without changing its DC.
void PhotoFrame(const fs::path& output, ConstImage3FView photo, size_t repeats,
                std::ofstream& timings) {
  thread_budget_internal::EncodeScope serial(1);
  const Extent2D padded{(photo.width() + 7) / 8 * 8, (photo.height() + 7) / 8 * 8};
  const Extent2D blocks{padded.width / 8, padded.height / 8};
  Image3FBuffer linear(padded), opsin(padded), reconstructed(photo.extent());
  for (size_t c = 0; c < 3; ++c) for (size_t y = 0; y < padded.height; ++y)
    for (size_t x = 0; x < padded.width; ++x)
      linear.view().plane[c].Row(y)[x] = photo.plane[c].Row(std::min(y, photo.height() - 1))[
          std::min(x, photo.width() - 1)];
  Check(LinearRgbToOpsin(linear.const_view(), 255.0f, opsin.view()));
  const size_t area = blocks.width * blocks.height;
  std::vector<float> initial(area), strategy_mask(area), pixel_mask(padded.width * padded.height),
      final_quant(area), distance(area);
  std::vector<double> scores;
  VarDctEncoderFrame frame;
  CpuQuantizationPipelineOptions options;
  options.fixed_dct8 = true;
  options.uniform_initial_quantization = true;
  options.adaptive_quantization.iterations = 0;
  options.adaptive_quantization.profile.loop_filter.gaborish = false;
  Check(RunCpuQuantizationPipeline(photo, opsin.const_view(), options, {
      .initial_quantization = {
          .quant_field = {initial.data(), blocks, blocks.width},
          .strategy_mask = {strategy_mask.data(), blocks, blocks.width},
          .pixel_mask = {pixel_mask.data(), padded, padded.width}},
      .adaptive_quantization = {
          .quant_field = {final_quant.data(), blocks, blocks.width},
          .block_distance_map = {distance.data(), blocks, blocks.width},
          .reconstructed_linear_rgb = reconstructed.view(), .frame = &frame,
          .score_history = &scores}}));
  for (auto prediction : {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted})
    RecordFrame(output, "photo_frame_p" + std::to_string(int(prediction)),
                vardct_frame_internal::BorrowFrame(frame), {.dc_prediction = prediction}, repeats, timings);
}

// Supplement the original serializer captures with exact numerical stage
// outputs, including the geometry where the Windows cross-compiler oracle
// differs on the frozen baseline. No expected values come from candidate code.
void DcStages(const fs::path& output) {
  if (!case_filter.empty()) return;
  Quantizer quantizer;
  Check(Quantizer::Create({799, 37}, &quantizer));
  for (auto extent : {Extent2D{1, 1}, {1, 257}, {257, 1}, {17, 19}, {255, 256}, {259, 259}}) {
    const size_t stride = extent.width + 7, area = stride * extent.height;
    std::array<std::vector<float>, 3> source, reconstructed;
    std::array<std::vector<int32_t>, 3> quantized;
    ConstImage3FView input;
    DcQuantizationOutput result;
    uint32_t random = 1729;
    for (size_t c = 0; c < 3; ++c) {
      source[c].assign(area, -900);
      reconstructed[c].assign(area, -901);
      quantized[c].assign(area, -777);
      for (size_t y = 0; y < extent.height; ++y) for (size_t x = 0; x < extent.width; ++x) {
        random = random * 1664525 + 1013904223;
        source[c][y * stride + x] = (static_cast<float>(random >> 16) - 32768) *
            quantizer.dc_steps()[c] * 0.17f;
      }
      input.plane[c] = {source[c].data(), extent, stride};
      result.quantized.plane[c] = {quantized[c].data(), extent, stride};
      result.reconstructed.plane[c] = {reconstructed[c].data(), extent, stride};
    }
    for (auto prediction : {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted})
      for (auto mode : {DcQuantizationMode::kRound, DcQuantizationMode::kPredictionAware})
        for (uint8_t precision = 0; precision < 4; ++precision) {
          Check(QuantizeDcCoefficients(input, quantizer, result, {mode, prediction, precision}));
          const auto name = "dc_" + std::to_string(extent.width) + "x" +
              std::to_string(extent.height) + "_p" + std::to_string(int(prediction)) +
              "_m" + std::to_string(int(mode)) + "_q" + std::to_string(precision);
          Snapshot record(output / (name + ".snapshot"));
          for (size_t c = 0; c < 3; ++c) for (size_t i = 0; i < area; ++i) {
            record.Number(std::bit_cast<uint32_t>(quantized[c][i]));
            record.Number(std::bit_cast<uint32_t>(reconstructed[c][i]));
          }
        }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 5) throw std::runtime_error("Usage: capture OUTPUT [REPEATS [PHOTO.pfm [CASE,...]]]");
    const fs::path output = argv[1];
    const size_t repeats = argc > 2 ? std::stoul(argv[2]) : 1;
    if (argc == 5) case_filter = argv[4];
    fs::create_directories(output);
    std::ofstream timings(output / "timings.csv");
    timings.exceptions(std::ios::badbit | std::ios::failbit);
    timings << "case,round,stage,nanoseconds\n";
    Synthetic(output, repeats, timings);
    Representations<int8_t>(output, repeats, timings);
    Representations<int16_t>(output, repeats, timings);
    Image3FBuffer image({65, 49});
    auto pixels = image.view();
    for (size_t c = 0; c < 3; ++c) for (size_t y = 0; y < 49; ++y)
      for (size_t x = 0; x < 65; ++x)
        pixels.plane[c].Row(y)[x] = float((x * 7 + y * 11 + c * 17) % 257) / 256;
    for (int effort : {1, 4, 7, 8, 9, 10}) {
      for (auto prediction : {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted}) {
        Workflow(output, "workflow_e" + std::to_string(effort) + "_p" + std::to_string(int(prediction)),
                 image.const_view(), {.effort = effort, .dc_prediction = prediction}, repeats, timings);
      }
    }
    Workflow(output, "workflow_target", image.const_view(),
        {.effort = 1, .rate_control_mode = VarDctRateControlMode::kTargetBytes,
         .target_bytes = 1500, .target_size_maximum_attempts = 3}, repeats, timings);
    if (argc >= 4) {
      Image3FBuffer photo;
      Check(io::ReadPfm(argv[3], &photo));
      Workflow(output, "workflow_photo", photo.const_view(), {.effort = 1}, repeats, timings);
      if (case_filter.find("photo_frame_") != std::string::npos)
        PhotoFrame(output, photo.const_view(), repeats, timings);
    }
    if (repeats == 0) DcStages(output);
    std::cout << "Captured Modular extraction fixtures in " << output.string() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
