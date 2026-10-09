// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/workflow.h"
#include "codestream/modular/search.h"
#include "codestream/modular/stream_encoder.h"
#include "io/pnm.h"
#ifdef GJXL_ORACLE_REFERENCE
#include "modular_reference.h"
#include "reference.h"
#endif
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <optional>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
// Every integer record is uint64 little endian; signed values use modulo 2^64.
// Never serialize object representation, pointers, size_t or host endianness.
struct Records {
  std::ofstream out;
  explicit Records(const fs::path &path) : out(path, std::ios::binary) {
    out.exceptions(std::ios::badbit | std::ios::failbit);
  }
  void Put(uint64_t v) {
    for (size_t i = 0; i < 8; ++i)
      out.put(static_cast<char>(v >> (8 * i)));
  }
  template <class... T> void Row(T... v) { (Put(static_cast<uint64_t>(v)), ...); }
};
void Bytes(const fs::path &path, std::span<const uint8_t> bytes) {
  std::ofstream out(path, std::ios::binary);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}
struct Fixture {
  std::string name;
  Extent2D extent;
  PackedModularFormat format;
  size_t channels, bits;
  std::vector<uint8_t> bytes;
  ModularCodingPolicy policy;
  PackedModularImageView view() const {
    return {bytes, extent, extent.width * channels * (bits / 8), format,
            SampleByteOrder::kLittleEndian};
  }
};
std::vector<Fixture> Corpus(const fs::path &photo, bool large) {
  std::vector<Fixture> result;
  const std::array<const char *, 7> names{"tiny",  "gray16",  "screen", "palette",
                                          "noise", "alpha16", "photo"};
  const std::array<PackedModularFormat, 7> formats{
      PackedModularFormat::kGray8, PackedModularFormat::kGray16, PackedModularFormat::kRgb8,
      PackedModularFormat::kRgba8, PackedModularFormat::kRgb16,  PackedModularFormat::kRgba16,
      PackedModularFormat::kRgb8};
  io::IntegerImage photographic;
  Ok(io::ReadPnm(photo, &photographic));
  Check(photographic.extent == Extent2D{125, 125}, "Unexpected photographic fixture");
  for (size_t kind = 0; kind < names.size(); ++kind) {
    const Extent2D extent = kind == 0   ? Extent2D{1, 1}
                            : kind == 6 ? (large ? Extent2D{625, 125} : Extent2D{125, 125})
                                        : (large ? Extent2D{513, 257} : Extent2D{257, 9});
    ModularInputProfile input;
    Ok(ResolveModularInput(extent, formats[kind], &input));
    Fixture f{names[kind], extent, formats[kind], input.channel_count, input.metadata.bits, {}, {}};
    uint32_t random = 0x718293ab;
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        for (size_t c = 0; c < f.channels; ++c) {
          random ^= random << 13;
          random ^= random >> 17;
          random ^= random << 5;
          uint32_t v = static_cast<uint32_t>(x * 257 + y * 129 + c * 7919);
          if (kind == 2 || kind == 3)
            v = static_cast<uint32_t>(((x / 7 + y / 3) % 16) * 13 + c * 37);
          if (kind == 4)
            v = random;
          if (kind == 6)
            v = photographic.bytes[(y * 125 + x % 125) * 3 + c];
          if (c == 3 && x % 3 == 0)
            v = 0;
          f.bytes.push_back(static_cast<uint8_t>(v));
          if (f.bits == 16)
            f.bytes.push_back(static_cast<uint8_t>(v >> 8));
        }
    if (kind == 1 || kind == 5) {
      f.policy.rct = kind == 5 ? 6 : 0;
      f.policy.transforms.size = 2;
      f.policy.transforms.entries[0] = {
          TransformKind::kSqueeze, 0, static_cast<uint8_t>(f.channels), 1, true, true};
      f.policy.transforms.entries[1] = {
          TransformKind::kSqueeze, 0, static_cast<uint8_t>(f.channels), 1, false, false};
    }
    if (kind == 3) {
      f.policy.transforms.size = 1;
      f.policy.transforms.entries[0] = {TransformKind::kPalette, 0, 4, 32};
    }
    if (kind == 2 || kind == 5 || kind == 6) {
      f.policy.tree.size = 3;
      f.policy.tree.nodes[0].property = 9;
      f.policy.tree.nodes[0].split = 64;
      f.policy.tree.nodes[0].left = 1;
      f.policy.tree.nodes[0].right = 2;
      f.policy.tree.nodes[1].predictor = Predictor::kWeighted;
      f.policy.tree.nodes[2].predictor = Predictor::kAverage4;
    }
    result.push_back(std::move(f));
  }
  return result;
}
void Policy(const fs::path &path, const ModularCodingPolicy &p) {
  Records r(path);
  r.Row(p.rct, p.transforms.size, p.tree.size);
  for (auto v : p.weighted.coefficients)
    r.Put(v);
  for (auto v : p.weighted.weights)
    r.Put(v);
  for (size_t i = 0; i < p.transforms.size; ++i) {
    auto t = p.transforms.entries[i];
    r.Row(t.kind, t.begin, t.count, t.colors, t.horizontal, t.in_place);
  }
  for (size_t i = 0; i < p.tree.size; ++i) {
    auto n = p.tree.nodes[i];
    r.Row(n.property, n.split, n.left, n.right, n.predictor, n.offset, n.multiplier);
  }
}
void Image(const fs::path &path, const ModularImage &image) {
  Records r(path);
  r.Row(image.channel_count(), image.metadata_channels());
  for (size_t c = 0; c < image.channel_count(); ++c) {
    auto v = image.view(c);
    auto d = v.descriptor;
    r.Row(d.extent.width, d.extent.height, d.hshift, d.vshift, d.role);
    for (size_t y = 0; y < d.extent.height; ++y)
      for (size_t x = 0; x < d.extent.width; ++x)
        r.Put(v.Row(y)[x]);
  }
}
void Decode(const Fixture &f, std::span<const uint8_t> bytes) {
#ifdef GJXL_ORACLE_REFERENCE
  auto decoded = test::modular_reference::DecodeLossless(bytes);
  Check(decoded.width == f.extent.width && decoded.height == f.extent.height &&
            decoded.channels == f.channels && decoded.bits == f.bits,
        "Reference profile mismatch");
  for (size_t i = 0; i < decoded.samples.size(); ++i) {
    uint16_t v = f.bytes[i * (f.bits / 8)];
    if (f.bits == 16)
      v |= static_cast<uint16_t>(f.bytes[i * 2 + 1]) << 8;
    Check(decoded.samples[i] == v, "Independent decoded sample mismatch");
  }
#else
  (void)f;
  (void)bytes;
#endif
}
void Freeze(const Fixture &f, EntropyCodingMode mode, const fs::path &root) {
  fs::create_directories(root);
  Bytes(root / "input.raw", f.bytes);
  Policy(root / "policy.bin", f.policy);
  Records description(root / "input.bin");
  description.Row(f.extent.width, f.extent.height, f.channels, f.bits, f.format, mode, 1);
  // Retain each prescribed transform prefix, including the pre-RCT source.
  ModularEncoderFrame frame;
  Ok(ModularEncoderFrame::Prepare(f.view(), &frame));
  Image(root / "source.bin", frame.image());
  for (size_t n = 0; n <= f.policy.transforms.size; ++n) {
    auto p = f.policy;
    p.transforms.size = n;
    Ok(ModularEncoderFrame::Prepare(f.view(), p, &frame));
    Image(root / ("transformed-" + std::to_string(n) + ".bin"), frame.image());
  }
  ModularWorkflowStoragePlan storage;
  Ok(ComputeModularWorkflowStoragePlan(f.extent, f.format, mode, f.policy, &storage));
  ChannelShape shape;
  Ok(DescribeImage(frame.image(), &shape));
  ModularStreamPlan layout;
  Ok(BuildModularStreamPlan(storage.geometry, shape.channels(), shape.metadata, &layout));
  PreparedModularTokens tokens;
  Ok(TokenizeModular(frame, layout, f.policy, &tokens));
#ifdef GJXL_ORACLE_REFERENCE
  modular_oracle_reference::Stages(f.view(), f.policy, frame.image(), layout, tokens);
#endif
  Records streams(root / "streams.bin"), decisions(root / "decisions.bin"),
      token_file(root / "tokens.bin");
  Records tree_tokens(root / "tree-tokens.bin");
  for (size_t i = 0; i < tokens.tree_token_count; ++i)
    tree_tokens.Row(tokens.tree_tokens[i].context, tokens.tree_tokens[i].value);
  TreeLayout tree;
  Ok(ValidateTree(f.policy.tree, &tree));
  size_t next = 0;
  for (size_t i = 0; i < layout.streams.size(); ++i) {
    auto s = layout.streams[i];
    streams.Row(s.role, s.id, s.section, s.slice_count, tokens.streams[i].size());
    for (auto slice : std::span(layout.slices).subspan(s.slice_begin, s.slice_count)) {
      auto rect = slice.rect;
      streams.Row(slice.channel, rect.x, rect.y, rect.extent.width, rect.extent.height);
      ModularChannelView v;
      Ok(BorrowChannelSlice(frame.image().view(slice.channel), rect, &v));
      std::optional<WeightedPredictor<resource_budget_internal::ResourceClass::kPreparation>> wp;
      if (tree.weighted)
        wp.emplace(rect.extent.width, f.policy.weighted);
      for (size_t y = 0; y < rect.extent.height; ++y) {
        int64_t previous = 0;
        for (size_t x = 0; x < rect.extent.width; ++x) {
          auto n = Neighbors(v, x, y);
          std::pair<int64_t, int64_t> weighted{};
          if (wp)
            weighted = wp->Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
          auto props = Properties(n, slice.channel, s.id, x, y, previous, weighted.second);
          previous = props[9];
          size_t leaf = Lookup(f.policy.tree, props);
          auto node = f.policy.tree.nodes[leaf];
          int64_t prediction = Predict(node.predictor, n, weighted.first);
          int64_t residual = int64_t{v.Row(y)[x]} - prediction - node.offset;
          decisions.Row(i, slice.channel, x, y, leaf, prediction, residual);
          for (auto p : props)
            decisions.Put(p);
          for (unsigned p = 0; p < 14; ++p)
            decisions.Put(Predict(static_cast<Predictor>(p), n, weighted.first));
          auto token = tokens.tokens[next++];
          token_file.Row(token.context, token.value);
          Check(token.context == tree.context[leaf] &&
                    token.value == PackSigned(static_cast<int32_t>(residual / node.multiplier)),
                "Trace/token mismatch");
          if (wp)
            Check(wp->Update(v.Row(y)[x], x, y), "Weighted overflow");
        }
      }
    }
  }
  Check(next == tokens.tokens.size(), "Trace coverage mismatch");
  EntropyCode model;
  Ok(OptimizeEntropyCode(tokens.streams,
                         {.context_count = static_cast<uint32_t>(tokens.context_count)}, &model));
  if (mode == EntropyCodingMode::kAns) {
    EntropyCode ans;
    Ok(OptimizeAnsEntropyCode(tokens.streams, model, &ans));
    model = std::move(ans);
  }
  Records model_file(root / "model.bin"), populations(root / "populations.bin"),
      bits(root / "stream-bits.bin");
  model_file.Row(model.mode, model.context_count, model.uint_configs.size(),
                 model.ans_log_alpha_size);
  for (auto v : model.context_map)
    model_file.Put(v);
  for (auto c : model.uint_configs)
    model_file.Row(c.split_exponent, c.msb_in_token, c.lsb_in_token);
  for (auto &p : model.prefix_codes) {
    model_file.Put(p.degenerate_symbol);
    for (auto v : p.depths)
      model_file.Put(v);
    for (auto v : p.bits)
      model_file.Put(v);
  }
  for (auto &h : model.ans_histograms) {
    model_file.Row(h.frequencies.size(), h.method, h.omit_position);
    for (auto v : h.frequencies)
      model_file.Put(v);
  }
  std::map<std::pair<uint32_t, uint32_t>, uint64_t> counts;
  for (auto t : tokens.tokens) {
    auto cluster = model.context_map[t.context];
    HybridUintToken h;
    Ok(EncodeHybridUint(t.value, model.uint_configs[cluster], &h));
    ++counts[{cluster, h.symbol}];
  }
  for (auto [key, count] : counts)
    populations.Row(key.first, key.second, count);
  BitWriter serialized_model;
  Ok(serialized_model.WithMaxBits(storage.maximum_global_bits, [&] {
    return WriteGlobalModelInTransaction(model, &serialized_model);
  }));
  Bytes(root / "model.raw", serialized_model.padded_bytes());
  bits.Put(serialized_model.bits_written());
  for (size_t i = 0; i < tokens.streams.size(); ++i) {
    BitWriter writer;
    if (tokens.streams[i].size())
      Ok(writer.WithMaxBits(storage.maximum_global_bits + storage.maximum_group_bits, [&] {
        return WriteStreamTokensWithValidatedModel(tokens.streams[i], model, &writer);
      }));
    bits.Row(i, writer.bits_written());
    Bytes(root / ("stream-" + std::to_string(i) + ".raw"), writer.padded_bytes());
  }
  std::vector<uint8_t> serial, parallel, profiled;
  gjxl::modular_internal::ModularEncodingOptions options;
  options.entropy = mode;
  options.coding = f.policy;
  Ok(EncodeModularImage(f.view(), options, &serial));
  ModularProfile profile;
  {
    ProfileSession session(&profile);
    Ok(EncodeModularImage(f.view(), options, &profiled));
  }
  Check(profiled == serial, "Profiling changed output");
  options.cpu_thread_count = 4;
  Ok(EncodeModularImage(f.view(), options, &parallel));
  Check(serial == parallel, "Serial/parallel mismatch");
  Decode(f, serial);
  Bytes(root / "final.jxl", serial);
  // Training is a separate policy baseline, not prescribed arithmetic truth.
  ModularCodingPolicy single, split;
  Ok(LearnModularPolicies(frame, layout, f.policy.rct, &single, &split));
  Policy(root / "learned-single.bin", single);
  Policy(root / "learned-split.bin", split);
}
void Benchmark(const Fixture &f, EntropyCodingMode mode, bool search, size_t threads,
               size_t repetitions, std::ostream &csv, const fs::path &root) {
  ModularWorkflowStoragePlan plan;
  Ok(search
         ? ComputeModularSearchStoragePlan(f.extent, f.format, mode, &plan, threads)
         : ComputeModularWorkflowStoragePlan(f.extent, f.format, mode, f.policy, &plan, threads));
  std::vector<uint8_t> expected;
  for (size_t run = 0; run <= repetitions; ++run) {
    gjxl::modular_internal::ModularEncodingOptions options;
    options.entropy = mode;
    options.search = search;
    if (!search)
      options.coding = f.policy;
    options.cpu_thread_count = threads;
    Ok(ExecutionDomain::Create(
        {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = threads},
        &options.execution_domain));
    std::vector<uint8_t> bytes;
    ModularProfile profile;
    auto start = Clock::now();
    {
      ProfileSession session(&profile);
      Ok(EncodeModularImage(f.view(), options, &bytes));
    }
    double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (run == 0) {
      expected = bytes;
      Decode(f, bytes);
      auto name = f.name + "-" + std::to_string(static_cast<int>(mode)) + "-" +
                  std::to_string(search) + "-" + std::to_string(threads);
      Policy(root / (name + "-resolved.bin"), profile.resolved);
      Bytes(root / (name + ".jxl"), bytes);
      // Replaying the observed winner must reproduce the searched complete file.
      auto replay = options;
      replay.search = false;
      replay.coding = profile.resolved;
      std::vector<uint8_t> prescribed;
      Ok(EncodeModularImage(f.view(), replay, &prescribed));
      Check(prescribed == bytes, "Resolved policy replay mismatch");
    } else
      Check(bytes == expected, "Benchmark output instability");
    auto s = options.execution_domain->snapshot();
    Check(s.peak_backing_bytes <= plan.working.peak_bytes && s.active_reservations == 0 &&
              s.active_cpu_participants == 0 && s.live_capacity_bytes == 0,
          "Benchmark resource leak/overrun");
    double stages = 0;
    for (double t : profile.seconds)
      stages += t;
    Check(stages <= seconds * 1.001, "Overlapping stage times");
    csv << f.name << ',' << f.extent.width << ',' << f.extent.height << ',' << f.channels << ','
        << f.bits << ",native," << static_cast<int>(mode) << ',' << search << ',' << threads << ','
        << run << ',' << seconds << ',' << bytes.size() << ',' << s.peak_backing_bytes << ','
        << plan.working.peak_bytes;
    for (double t : profile.seconds)
      csv << ',' << t;
    csv << ',' << profile.encoded_candidates << '\n';
  }
#ifdef GJXL_ORACLE_REFERENCE
  if (threads == 1 && !search) {
    test::modular_reference::IntegerImage image{f.extent.width,
                                                f.extent.height,
                                                static_cast<uint32_t>(f.channels),
                                                static_cast<uint32_t>(f.bits),
                                                {}};
    for (size_t i = 0; i < f.bytes.size(); i += f.bits / 8)
      image.samples.push_back(
          static_cast<uint16_t>(f.bytes[i] | (f.bits == 16 ? uint32_t{f.bytes[i + 1]} << 8 : 0)));
    for (size_t run = 0; run <= repetitions; ++run) {
      auto start = Clock::now();
      auto bytes = test::modular_reference::EncodeLossless(image);
      double seconds = std::chrono::duration<double>(Clock::now() - start).count();
      Decode(f, bytes);
      csv << f.name << ',' << f.extent.width << ',' << f.extent.height << ',' << f.channels << ','
          << f.bits << ",libjxl-effort1,-1,0,1," << run << ',' << seconds << ',' << bytes.size()
          << ",0,0,0,0,0,0,0,0,0,0\n";
    }
  }
#endif
}
} // namespace
int main(int argc, char **argv) try {
  if (argc < 4 || argc > 5)
    throw std::runtime_error(
        "Usage: capture freeze|benchmark PHOTO.ppm OUTPUT [warm-repetitions=3]");
  const bool bench = std::string(argv[1]) == "benchmark";
  Check(bench || std::string(argv[1]) == "freeze", "Invalid command");
  size_t repetitions = argc == 5 ? std::stoul(argv[4]) : 3;
  Check(repetitions > 0 && repetitions <= 100, "Invalid repetitions");
  fs::path output = argv[3];
  fs::create_directories(output);
  std::ofstream csv(output / "timings.csv");
  csv.exceptions(std::ios::badbit | std::ios::failbit);
  csv << std::setprecision(17)
      << "fixture,width,height,channels,bits,encoder,entropy,search,threads,run,seconds,bytes,"
         "managed_peak,planned_capacity,input,transforms,training,tokens,model,emission,assembly,"
         "candidates\n";
  for (const auto &f : Corpus(argv[2], bench))
    for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
      std::cout << f.name << ' ' << static_cast<int>(mode) << std::endl;
      if (bench)
        for (bool search : {false, true})
          for (size_t threads : {size_t{1}, size_t{4}})
            Benchmark(f, mode, search, threads, repetitions, csv, output);
      else
        Freeze(f, mode,
               output / (f.name + (mode == EntropyCodingMode::kPrefix ? "-prefix" : "-ans")));
    }
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
