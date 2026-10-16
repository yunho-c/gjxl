// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "native_fit.h"
#include "native_preparation.h"
#include "paired_learning.h"
#include "policy_trace.h"
#include "reference_capture.h"
#include "stream_replay.h"
#include "coding_replay.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

#include "codestream/modular/search.h"
#include "codestream/modular/tokenization.h"
#include "codestream/modular/workflow.h"
#include "io/pnm.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/transform/transform.h"
#include "modular_reference.h"

namespace native = gjxl::modular_internal;
namespace codec = gjxl::test::modular_reference;
namespace capture = reference_learning;

namespace {
void Require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const gjxl::Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
void WriteBytes(const std::filesystem::path &path,
                const std::vector<uint8_t> &bytes) {
  Require(!std::filesystem::exists(path),
          "Refusing to overwrite an encoded output");
  std::ofstream stream(path, std::ios::binary);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  stream.close();
}
codec::IntegerImage Unpack(const gjxl::io::IntegerImage &input,
                           const native::ModularInputProfile &profile) {
  codec::IntegerImage original{input.extent.width,
                               input.extent.height,
                               static_cast<uint32_t>(profile.channel_count),
                               profile.metadata.bits,
                               {}};
  for (size_t y = 0; y < input.extent.height; ++y) {
    for (size_t x = 0; x < input.extent.width * profile.channel_count; ++x) {
      size_t pos = y * input.row_stride + x * profile.bytes_per_sample;
      original.samples.push_back(profile.bytes_per_sample == 1
                                     ? input.bytes[pos]
                                     : (uint16_t{input.bytes[pos]} << 8) |
                                           input.bytes[pos + 1]);
    }
  }
  return original;
}
native::ModularCodingPolicy ReadPolicy(const std::filesystem::path &path) {
  native::ModularCodingPolicy policy;
  std::ifstream in(path);
  in.exceptions(std::ios::badbit | std::ios::failbit);
  unsigned rct;
  in >> rct >> policy.transforms.size >> policy.tree.size;
  Require(rct < 42 && policy.transforms.size <= native::kMaximumTransforms &&
              policy.tree.size && policy.tree.size <= native::kMaximumTreeNodes,
          "Invalid saved policy dimensions");
  policy.rct = rct;
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    auto &t = policy.transforms.entries[i];
    unsigned kind, begin, count;
    in >> kind >> begin >> count >> t.colors >> t.horizontal >> t.in_place;
    Require(kind <= 1 && begin <= 255 && count <= 255,
            "Invalid saved transform");
    t.kind = static_cast<native::TransformKind>(kind);
    t.begin = begin;
    t.count = count;
  }
  policy.tree.nodes.resize(policy.tree.size);
  for (size_t i = 0; i < policy.tree.size; ++i) {
    auto &n = policy.tree.nodes[i];
    unsigned predictor;
    in >> n.property >> n.split >> n.left >> n.right >> predictor >> n.offset >>
        n.multiplier;
    Require(predictor < 14, "Invalid saved predictor");
    n.predictor = static_cast<native::Predictor>(predictor);
  }
  Ok(native::ValidateCodingPolicy(policy));
  return policy;
}
void WritePolicy(const std::filesystem::path &path,
                 const native::ModularCodingPolicy &policy) {
  std::ofstream out(path);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out << unsigned(policy.rct) << ' ' << policy.transforms.size << ' '
      << policy.tree.size << '\n';
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    const auto &t = policy.transforms.entries[i];
    out << unsigned(t.kind) << ' ' << unsigned(t.begin) << ' '
        << unsigned(t.count) << ' ' << t.colors << ' ' << t.horizontal << ' '
        << t.in_place << '\n';
  }
  for (size_t i = 0; i < policy.tree.size; ++i) {
    const auto &n = policy.tree.nodes[i];
    out << n.property << ' ' << n.split << ' ' << n.left << ' ' << n.right
        << ' ' << unsigned(n.predictor) << ' ' << n.offset << ' '
        << n.multiplier << '\n';
  }
  out.close();
}

// Copy each canonical stream's local slices, retaining holes in stream IDs.
// Coordinates and predictors reset at each slice exactly as in native coding.
std::vector<jxl::Image>
ReferenceImages(const native::ModularEncoderFrame &frame,
                const native::ModularStreamPlan &plan,
                JxlMemoryManager *manager) {
  size_t count = 0;
  for (const auto &stream : plan.streams)
    count = std::max(count, stream.id + 1);
  Require(count <= std::numeric_limits<uint32_t>::max(),
          "Unsupported stream count");
  std::vector<jxl::Image> images;
  images.reserve(count);
  for (size_t i = 0; i < count; ++i)
    images.emplace_back(manager);
  for (const auto &stream : plan.streams) {
    if (stream.slice_count == 0)
      continue;
    auto &image = images[stream.id];
    image.w = frame.metadata().extent.width;
    image.h = frame.metadata().extent.height;
    image.bitdepth = frame.metadata().bits;
    image.error = false;
    for (const auto &slice :
         std::span(plan.slices)
             .subspan(stream.slice_begin, stream.slice_count)) {
      native::ModularChannelView view;
      Ok(native::BorrowChannelSlice(frame.image().view(slice.channel),
                                    slice.rect, &view));
      const auto &d = view.descriptor;
      auto created = jxl::Channel::Create(manager, d.extent.width,
                                          d.extent.height, d.hshift, d.vshift);
      Require(created.ok(), "Reference slice allocation failed");
      auto channel = std::move(created).value_();
      for (size_t y = 0; y < d.extent.height; ++y)
        std::copy_n(view.Row(y), d.extent.width, channel.Row(y));
      image.channel.push_back(std::move(channel));
      if (d.role == native::ChannelRole::kMetadata) {
        Require(image.nb_meta_channels + 1 == image.channel.size(),
                "Metadata is not a leading channel");
        ++image.nb_meta_channels;
      }
    }
  }
  return images;
}

jxl::ModularOptions ReferenceOptions(const native::ModularCodingPolicy &policy,
                                     bool all_predictors) {
  // enc_modular.cc: lossless e9 (kTortoise), decode tier 0, learning fraction
  // 0.5 * 1.3, all non-reference properties, default WP header. These settings
  // are fixed controls; no capacity-driven threshold search is performed.
  Require(policy.weighted == native::WeightedPredictorParameters{},
          "Unsupported custom WP header in research adapter");
  bool squeeze = false;
  for (size_t i = 0; i < policy.transforms.size; ++i)
    squeeze |=
        policy.transforms.entries[i].kind == native::TransformKind::kSqueeze;
  jxl::ModularOptions options;
  options.predictor =
      all_predictors ? jxl::Predictor::Variable : jxl::Predictor::Best;
  options.nb_repeats = 0.5f * 1.3f;
  options.max_property_values = 256;
  options.splitting_heuristics_node_threshold = 89;
  options.fast_decode_multiplier = 1.001f;
  options.splitting_heuristics_properties =
      squeeze ? std::vector<uint32_t>{0, 1,  4,  5,  6,  7,  8, 15,
                                      9, 10, 11, 12, 13, 14, 2, 3}
              : std::vector<uint32_t>{0,  1, 15, 9, 10, 11, 12, 13,
                                      14, 2, 3,  4, 5,  6,  7,  8};
  return options;
}

native::DecisionTree ImportTree(const jxl::Tree &source,
                                uint64_t source_samples) {
  Require(!source.empty() && source.size() <= native::kMaximumTreeNodes,
          "Reference tree exceeds native representation");
  native::DecisionTree result;
  result.size = source.size();
  result.nodes.resize(result.size);
  for (size_t i = 0; i < source.size(); ++i) {
    const auto &n = source[i];
    Require(n.property >= -1 &&
                n.property < static_cast<int>(native::kPropertyCount) &&
                n.predictor_offset >= INT32_MIN &&
                n.predictor_offset <= INT32_MAX && n.multiplier &&
                n.multiplier <= INT32_MAX &&
                static_cast<unsigned>(n.predictor) < 14,
            "Reference node is outside native profile");
    auto &out = result.nodes[i];
    out.property = n.property;
    out.split = n.splitval;
    out.predictor = static_cast<native::Predictor>(n.predictor);
    out.offset = static_cast<int32_t>(n.predictor_offset);
    out.multiplier = n.multiplier;
    if (n.property != -1) {
      Require(n.lchild < source.size() && n.rchild < source.size(),
              "Invalid reference child");
      out.left = n.lchild;
      out.right = n.rchild;
    }
  }
  native::TreeLayout layout;
  Ok(native::ValidateTree(result, &layout));
  Require(layout.leaves <= native::MaximumGlobalTreeLeaves(source_samples),
          "Reference tree exceeds native image-dependent limit");
  return result;
}

void CheckTokens(const native::ModularEncoderFrame &frame,
                 const native::ModularStreamPlan &plan,
                 const native::ModularCodingPolicy &policy,
                 const std::vector<jxl::Image> &images,
                 const std::vector<jxl::ModularOptions> &options,
                 const jxl::Tree &raw_tree) {
  native::PreparedModularTokens tokens;
  Ok(native::TokenizeModular(frame, plan, policy, &tokens));
  std::vector<jxl::Token> tree_tokens;
  jxl::Tree decoder_tree;
  Require(bool(jxl::TokenizeTree(raw_tree, &tree_tokens, &decoder_tree)),
          "Reference tree tokenization failed");
  Require(tokens.tree_token_count == tree_tokens.size(),
          "Tree token count mismatch");
  for (size_t i = 0; i < tree_tokens.size(); ++i) {
    Require(tokens.tree_tokens[i].context == tree_tokens[i].context &&
                tokens.tree_tokens[i].value == tree_tokens[i].value,
            "Tree token mismatch");
  }
  for (size_t i = 0; i < plan.streams.size(); ++i) {
    const size_t id = plan.streams[i].id;
    std::vector<jxl::Token> reference;
    jxl::GroupHeader header;
    size_t width = 0;
    Require(bool(jxl::ModularCompress(images[id], options[id], id, decoder_tree,
                                      header, reference, &width)),
            "Reference tokenization failed");
    Require(tokens.streams[i].size() == reference.size(),
            "Stream token count mismatch");
    for (size_t j = 0; j < reference.size(); ++j) {
      const auto token = tokens.streams[i][j];
      if (token.context != reference[j].context ||
          token.value != reference[j].value) {
        throw std::runtime_error("Stream token mismatch at stream " +
                                 std::to_string(id) + " token " +
                                 std::to_string(j));
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) try {
  std::cout << std::setprecision(17);
  Require(argc >= 2, "Usage: reference_learning reference|bridge|replay ...");
  const std::string mode = argv[1];
  if (mode == "stream-cost-check" || mode == "stream-wp-cost-check") {
    Require(argc == 4, "stream-[wp-]cost-check SOURCE DESCRIPTION");
    if (mode == "stream-cost-check")
      capture::ReplaySelectionCostInput(argv[2], argv[3], std::cout);
    else
      capture::ReplayWeightedCostInput(argv[2], argv[3], std::cout);
    return 0;
  }
  if (mode == "stream-entropy-check" || mode == "stream-entropy-time") {
    Require(argc == 6, "stream-entropy-check SOURCE DESCRIPTION OUTPUT VARIANTS");
    capture::ReplayEntropyInput(argv[2], argv[3], argv[4], argv[5], std::cout,
                               mode == "stream-entropy-time");
    return 0;
  }
  if (mode == "stream-encode-check") {
    Require(argc == 5, "stream-encode-check SOURCE DESCRIPTION OUTPUT");
    capture::ReplayCodingInput(argv[2], argv[3], argv[4], std::cout);
    return 0;
  }
  if (mode == "stream-prepare-check") {
    Require(argc == 4, "stream-prepare-check DESCRIPTION PIXELS");
    capture::ReplayLearningInput(argv[2], argv[3], std::cout);
    return 0;
  }
  if (mode == "validation-replay") {
    Require(argc == 4, "validation-replay FIXTURE REPEATS");
    const size_t repeats = std::stoul(argv[3]);
    Require(repeats && repeats <= 10000, "Invalid validation repetitions");
    const auto fixture = capture::ReadFixture(argv[2]);
    const auto samples = capture::NativeSamples(fixture.samples);
    std::vector<native::HistogramMultiplierRange> multipliers;
    for (const auto &m : fixture.multipliers)
      multipliers.push_back({m.range, m.multiplier});
    native::HistogramTreeLearningOptions options;
    options.split_threshold = fixture.threshold;
    options.fast_decode_multiplier = fixture.fast_decode_multiplier;
    options.static_range = fixture.range;
    options.multipliers = multipliers;
    options.maximum_samples = native::kHistogramMaximumSamples;
    Ok(native::ValidateHistogramTrainingSamples(samples, options));
    const capture::Timer timer;
    for (size_t i = 0; i < repeats; ++i)
      Ok(native::ValidateHistogramTrainingSamples(samples, options));
    const auto elapsed = timer.Elapsed();
    std::cout << "{\"repeats\":" << repeats << ",\"cpu\":" << elapsed.cpu
              << ",\"wall\":" << elapsed.wall << ",\"validated\":true}\n";
    return 0;
  }
  if (mode == "preparation-self-test") {
    Require(argc == 2, "preparation-self-test takes no arguments");
    const auto count = capture::RunPreparationDifferentialTests();
    std::cout << "{\"preparation_differential_cases\":" << count
              << ",\"unique_proof_variants\":2"
              << ",\"samples_exact\":true,\"tree_exact\":true,\"reference_"
                 "weight_overflow_rejected\":true}\n";
    return 0;
  }
  if (mode == "prepare-check" || mode == "paired-learning") {
    const bool paired = mode == "paired-learning";
    Require(paired ? argc == 7 : (argc == 6 || argc == 7),
            "prepare-check INPUT POLICY FIXTURE best|all [OUTPUT]; "
            "paired-learning INPUT POLICY FIXTURE best|all REPEATS");
    Require(std::string(argv[5]) == "best" || std::string(argv[5]) == "all",
            "Invalid predictor control");
    gjxl::io::IntegerImage input;
    Ok(gjxl::io::ReadPnm(argv[2], &input));
    native::PackedModularImageView view{
        input.bytes, input.extent, input.row_stride,
        static_cast<native::PackedModularFormat>(input.format),
        native::SampleByteOrder::kBigEndian};
    auto policy = ReadPolicy(argv[3]);
    native::ModularEncoderFrame frame;
    Ok(native::ModularEncoderFrame::Prepare(view, policy, &frame));
    native::ChannelShape shape;
    Ok(native::DescribeImage(frame.image(), &shape));
    native::ModularFrameGeometry geometry;
    Ok(native::ModularFrameGeometry::Create(input.extent, &geometry));
    native::ModularStreamPlan plan;
    Ok(native::BuildModularStreamPlan(geometry, shape.channels(),
                                      shape.metadata, &plan));
    const auto fixture = capture::ReadFixture(argv[4]);
    if (paired) {
      JxlMemoryManager manager;
      Require(bool(jxl::MemoryManagerInit(&manager, nullptr)),
              "Reference memory manager initialization failed");
      const auto images = ReferenceImages(frame, plan, &manager);
      capture::RunPairedLearning(
          images, ReferenceOptions(policy, std::string(argv[5]) == "all"),
          fixture, std::stoul(argv[6]), std::cout);
      return 0;
    }
    size_t count = 0;
    for (const auto &stream : plan.streams)
      count = std::max(count, stream.id + 1);
    capture::BorrowedTrainingImages borrowed;
    borrowed.channels.resize(count);
    borrowed.images.resize(count);
    for (const auto &stream : plan.streams) {
      auto &channels = borrowed.channels[stream.id];
      size_t metadata = 0;
      for (const auto &slice :
           std::span(plan.slices)
               .subspan(stream.slice_begin, stream.slice_count)) {
        native::ModularChannelView channel;
        Ok(native::BorrowChannelSlice(frame.image().view(slice.channel),
                                      slice.rect, &channel));
        channels.push_back(channel);
        metadata += channel.descriptor.role == native::ChannelRole::kMetadata;
      }
      if (!channels.empty())
        borrowed.images[stream.id] = {input.extent, channels, metadata};
    }
    const auto options = capture::NativeSamplingOptions(
        ReferenceOptions(policy, std::string(argv[5]) == "all"));
    using namespace gjxl::resource_budget_internal;
    HostStorageBound bound;
    Ok(native::ComputeHistogramSamplePreparationStorageBound(borrowed.images,
                                                             options, &bound));
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    native::PreparedHistogramSamples prepared;
    capture::Timing preparation;
    {
      const ResourceContextScope scope(
          {&reservation, ResourceClass::kPreparation});
      const capture::Timer timer;
      Ok(native::PrepareHistogramSamples(borrowed.images, &prepared, options));
      preparation = timer.Elapsed();
    }
    const auto peak = budget.snapshot().peak_backing_bytes;
    capture::CheckPreparedSamples(prepared, fixture);
    policy.tree = capture::CheckPreparedTree(prepared, fixture.expected);
    std::vector<uint8_t> encoded;
    if (argc == 7) {
      native::ModularEncodingOptions encoding;
      encoding.coding = policy;
      encoding.entropy = gjxl::EntropyCodingMode::kAns;
      encoding.ans_selection = gjxl::AnsSelectionMode::kEstimated;
      encoding.cpu_thread_count = 4;
      Ok(native::EncodeModularImage(view, encoding, &encoded));
      native::ModularInputProfile profile;
      Ok(native::ResolveModularInput(input.extent, view.format, &profile));
      Require(codec::DecodeLossless(encoded) == Unpack(input, profile),
              "Native prepared reconstruction failed");
      WriteBytes(argv[6], encoded);
    }
    const auto &stats = prepared.stats();
    std::cout << "{\"samples_exact\":true,\"tree_exact\":true,\"samples\":"
              << stats.selected_samples
              << ",\"distinct\":" << prepared.samples().weights.size()
              << ",\"quantization_samples\":" << stats.quantization_samples
              << ",\"scanned_pixels\":" << stats.scanned_pixels
              << ",\"population\":" << stats.population
              << ",\"channels\":" << stats.channels
              << ",\"preparation_cpu\":" << preparation.cpu
              << ",\"preparation_wall\":" << preparation.wall
              << ",\"storage_bound_bytes\":" << bound.peak_bytes
              << ",\"storage_peak_bytes\":" << peak
              << ",\"bytes\":" << encoded.size()
              << ",\"decoded_exact\":" << (argc == 7 ? "true" : "null")
              << "}\n";
    return 0;
  }
  if (mode == "native-self-test") {
    Require(argc == 2, "native-self-test takes no arguments");
    const auto count = capture::RunNativeDifferentialTests();
    std::cout << "{\"native_differential_cases\":" << count
              << ",\"tree_exact\":true}\n";
    return 0;
  }
  if (mode == "native-replay" || mode == "native-verify" ||
      mode == "paired-replay") {
    const bool verify = mode == "native-verify";
    const bool paired = mode == "paired-replay";
    Require(
        argc == (verify ? 3 : 4),
        "native-verify FIXTURE; native-replay|paired-replay FIXTURE REPEATS");
    const size_t repeats = verify ? 1 : std::stoul(argv[3]);
    Require(repeats && repeats <= 100, "Invalid replay repetitions");
    const auto fixture = capture::ReadFixture(argv[2]);
    size_t measured_peak = 0;
    if (!verify) {
      if (paired)
        (void)capture::Replay(fixture);
      measured_peak = capture::FitNative(fixture, true).scratch_peak_bytes;
    }
    for (size_t i = 0; i < repeats; ++i) {
      capture::Timing reference;
      if (paired && i % 2 == 0)
        reference = capture::Replay(fixture);
      const auto result = capture::FitNative(fixture, verify);
      if (paired && i % 2 != 0)
        reference = capture::Replay(fixture);
      std::cout << "{\"iteration\":" << i
                << ",\"tree_exact\":true,\"nodes\":" << result.tree.size
                << ",\"samples\":" << fixture.samples.NumSamples()
                << ",\"distinct\":" << fixture.samples.NumDistinctSamples()
                << ",\"predictors\":" << fixture.samples.NumPredictors()
                << ",\"properties\":" << fixture.samples.NumProperties()
                << ",\"cpu\":" << result.time.cpu
                << ",\"wall\":" << result.time.wall
                << ",\"scratch_bound_bytes\":" << result.scratch_bound_bytes
                << ",\"sample_backing_bytes\":" << result.sample_backing_bytes;
      std::cout << ",\"scratch_peak_bytes\":"
                << (verify ? result.scratch_peak_bytes : measured_peak);
      if (paired)
        std::cout << ",\"reference_cpu\":" << reference.cpu
                  << ",\"reference_wall\":" << reference.wall
                  << ",\"reference_first\":" << (i % 2 == 0 ? "true" : "false");
      std::cout << "}\n";
    }
    return 0;
  }
  if (mode == "rewrite") {
    Require(argc == 4, "rewrite INPUT_FIXTURE OUTPUT_FIXTURE");
    capture::WriteFixture(argv[3], capture::ReadFixture(argv[2]));
    return 0;
  }
  if (mode == "self-test") {
    Require(argc == 2, "self-test takes no arguments");
    jxl::Tree good{
        jxl::PropertyDecisionNode::Split(9, -7, 1),
        jxl::PropertyDecisionNode::Leaf(jxl::Predictor::Weighted),
        jxl::PropertyDecisionNode::Leaf(jxl::Predictor::Gradient, -3, 2)};
    const auto imported = ImportTree(good, 1000);
    Require(imported.nodes[0].left == 1 && imported.nodes[0].right == 2 &&
                imported.nodes[0].split == -7 &&
                imported.nodes[2].offset == -3 &&
                imported.nodes[2].multiplier == 2,
            "Valid import fields changed");
    auto rejects = [&](auto mutate) {
      auto bad = good;
      mutate(bad);
      bool rejected = false;
      try {
        (void)ImportTree(bad, 1000);
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      Require(rejected, "Invalid reference tree was accepted");
    };
    rejects([](auto &t) { t.clear(); });
    rejects([](auto &t) { t[0].property = 16; });
    rejects([](auto &t) { t[0].lchild = 3; });
    rejects([](auto &t) { t[0].lchild = 0; });
    rejects([](auto &t) { t[0].rchild = t[0].lchild; });
    rejects([](auto &t) { t[1].predictor_offset = INT64_MAX; });
    rejects([](auto &t) { t[1].predictor = jxl::Predictor::Best; });
    rejects([](auto &t) { t[1].multiplier = 0; });
    rejects([](auto &t) { t[1].multiplier = UINT32_MAX; });
    std::cout << "{\"import_checks\":10,\"passed\":true}\n";
    return 0;
  }
  if (mode == "replay") {
    Require(argc == 4, "replay FIXTURE REPEATS");
    const size_t repeats = std::stoul(argv[3]);
    Require(repeats && repeats <= 100, "Invalid replay repetitions");
    auto fixture = capture::ReadFixture(argv[2]);
    (void)capture::Replay(
        fixture); // Warmup, including exact tree verification.
    for (size_t i = 0; i < repeats; ++i) {
      const auto t = capture::Replay(fixture);
      std::cout << "{\"iteration\":" << i
                << ",\"tree_exact\":true,\"nodes\":" << fixture.expected.size()
                << ",\"cpu\":" << t.cpu << ",\"wall\":" << t.wall << "}\n";
    }
    return 0;
  }
  const bool reference = mode == "reference" || mode == "reference-policy";
  Require((mode == "reference" && argc == 5) || (mode == "reference-policy" && argc == 6) || (mode == "bridge" && argc == 7),
          "reference INPUT OUT capture|timing; reference-policy INPUT OUT trace|timing auto|buffered; bridge INPUT POLICY OUT "
          "capture|timing best|all");
  const std::filesystem::path directory = argv[reference ? 3 : 4];
  const std::string recording = argv[reference ? 4 : 5];
  Require(recording == "capture" || recording == "timing" || (recording == "trace" && mode == "reference-policy"),
          "Invalid recording mode");
  if (mode == "reference-policy") {
    Require(recording != "capture", "Use stream trace for policy captures");
    const std::string buffering = argv[5];
    Require(buffering == "auto" || buffering == "buffered", "Invalid buffering control");
    capture::reference_buffering = buffering == "auto" ? -1 : 0;
  }
  std::filesystem::create_directories(directory);
  Require(!std::filesystem::exists(directory / "output.jxl"),
          "Output already exists");
  capture::Session session;
  if (recording == "trace") {
    session.stream_directory = directory / "streams";
    std::filesystem::create_directories(session.stream_directory);
  }
  if (recording == "capture") {
    session.directory = directory / "fixtures";
    std::filesystem::create_directories(session.directory);
  }
  gjxl::io::IntegerImage input;
  Ok(gjxl::io::ReadPnm(argv[2], &input));
  native::PackedModularImageView view{
      input.bytes, input.extent, input.row_stride,
      static_cast<native::PackedModularFormat>(input.format),
      native::SampleByteOrder::kBigEndian};
  native::ModularInputProfile profile;
  Ok(native::ResolveModularInput(input.extent, view.format, &profile));
  const auto original = Unpack(input, profile);
  std::vector<uint8_t> encoded;
  bool token_exact = false;
  capture::Timing elapsed, preparation;
  if (reference) {
    const capture::ScopedSession scoped(&session);
    const capture::Timer timer;
    encoded = codec::EncodeLossless(original);
    elapsed = timer.Elapsed();
  } else {
    Require(std::string(argv[6]) == "best" || std::string(argv[6]) == "all",
            "Invalid predictor control");
    auto policy = ReadPolicy(argv[3]);
    const capture::Timer prep_timer;
    native::ModularEncoderFrame frame;
    Ok(native::ModularEncoderFrame::Prepare(view, policy, &frame));
    native::ChannelShape shape;
    Ok(native::DescribeImage(frame.image(), &shape));
    native::ModularFrameGeometry geometry;
    Ok(native::ModularFrameGeometry::Create(input.extent, &geometry));
    native::ModularStreamPlan plan;
    Ok(native::BuildModularStreamPlan(geometry, shape.channels(),
                                      shape.metadata, &plan));
    JxlMemoryManager manager;
    Require(bool(jxl::MemoryManagerInit(&manager, nullptr)),
            "Reference memory manager initialization failed");
    auto images = ReferenceImages(frame, plan, &manager);
    auto option = ReferenceOptions(policy, std::string(argv[6]) == "all");
    std::vector<jxl::ModularOptions> options(images.size(), option);
    preparation = prep_timer.Elapsed();
    jxl::Tree tree;
    {
      const capture::ScopedSession scoped(&session);
      const capture::Timer timer;
      auto learned =
          jxl::LearnTree(images.data(), options.data(), 0, images.size());
      Require(learned.ok(), "Reference learning failed");
      tree = std::move(learned).value_();
      elapsed = timer.Elapsed();
    }
    policy.tree = ImportTree(tree, original.samples.size());
    CheckTokens(frame, plan, policy, images, options, tree);
    token_exact = true;
    native::ModularEncodingOptions encoding;
    encoding.coding = policy;
    encoding.entropy = gjxl::EntropyCodingMode::kAns;
    encoding.ans_selection = gjxl::AnsSelectionMode::kEstimated;
    encoding.cpu_thread_count = 4;
    Ok(native::EncodeModularImage(view, encoding, &encoded));
    WritePolicy(directory / "policy.txt", policy);
  }
  Require(codec::DecodeLossless(encoded) == original,
          "Exact reconstruction failed");
  WriteBytes(directory / "output.jxl", encoded);
  rusage usage{};
  Require(getrusage(RUSAGE_SELF, &usage) == 0,
          "Cannot read process resource usage");
  const uint64_t peak_rss =
      uint64_t{static_cast<unsigned long>(usage.ru_maxrss)}
#if !defined(__APPLE__)
      * 1024
#endif
      ;
  std::cout << "{\"mode\":\"" << mode << "\",\"bytes\":" << encoded.size()
            << ",\"buffering\":" << capture::reference_buffering
            << ",\"traced_learning_calls\":" << session.traced_learning_calls
            << ",\"traced_coding_streams\":" << session.traced_coding_streams
            << ",\"decoded_exact\":true,\"tokens_exact\":"
            << (token_exact ? "true" : "null")
            << ",\"process_peak_rss_bytes\":" << peak_rss
            << ",\"reference_call_cpu\":" << elapsed.cpu
            << ",\"reference_call_wall\":" << elapsed.wall
            << ",\"bridge_preparation_cpu\":" << preparation.cpu
            << ",\"bridge_preparation_wall\":" << preparation.wall
            << ",\"learning\":";
  capture::PrintSession(std::cout, session);
  std::cout << "}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
