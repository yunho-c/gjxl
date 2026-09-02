// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>

#if GJXL_LIBJXL_STAGE_PROFILE
#include "lib/jxl/enc_stage_profile.h"
#endif

#ifndef GJXL_LIBJXL_REVISION
#error "GJXL_LIBJXL_REVISION must identify the linked libjxl checkout"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::filesystem::path input;
  std::filesystem::path raw_samples;
  std::filesystem::path output;
  float distance = 1.0f;
  int effort = 7;
  size_t num_threads = 0;
  size_t warmups = 2;
  size_t samples = 5;
  bool stage_profile = false;
};

struct PfmImage {
  size_t width = 0;
  size_t height = 0;
  std::vector<float> pixels;
};

struct Sample {
  size_t index = 0;
  uint64_t elapsed_nanoseconds = 0;
  size_t encoded_bytes = 0;
#if GJXL_LIBJXL_STAGE_PROFILE
  jxl::EncoderStageProfileSink stage_profile;
#endif
};

#if GJXL_LIBJXL_STAGE_PROFILE
constexpr std::array<std::string_view, jxl::kEncoderProfilePhaseCount>
    kPhaseNames = {"coefficient_tokenization", "entropy_model_construction",
                   "model_and_token_emission", "framing_and_assembly",
                   "complete_serializer"};
constexpr std::array<std::string_view, jxl::kEncoderProfileWorkCount>
    kWorkNames = {
        "coefficient_tokenization",       "histogram_population",
        "histogram_clustering",           "hybrid_uint_selection",
        "entropy_model_construction",     "histogram_serialization",
        "token_encoding_and_bit_writing", "modular_and_dc_side_data_encoding",
        "output_assembly_and_copying"};
constexpr std::array<std::string_view, jxl::kEncoderProfileCountCount>
    kCountNames = {"token_count", "histogram_count", "model_bits", "token_bits",
                   "output_bytes"};
#endif

[[nodiscard]] uint32_t ByteSwap32(uint32_t value) noexcept {
  return ((value & 0x000000ffu) << 24) | ((value & 0x0000ff00u) << 8) |
         ((value & 0x00ff0000u) >> 8) | ((value & 0xff000000u) >> 24);
}

[[nodiscard]] std::string ReadHeaderLine(std::ifstream *input) {
  std::string line;
  while (std::getline(*input, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (!line.empty() && line.front() != '#')
      return line;
  }
  throw std::runtime_error("PFM header is incomplete");
}

[[nodiscard]] PfmImage ReadPfm(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open PFM input: " + path.string());
  }
  if (ReadHeaderLine(&input) != "PF") {
    throw std::runtime_error("Comparison input must be a three-channel PFM");
  }

  PfmImage image;
  std::string trailing;
  {
    std::istringstream dimensions(ReadHeaderLine(&input));
    if (!(dimensions >> image.width >> image.height) ||
        (dimensions >> trailing) || image.width == 0 || image.height == 0) {
      throw std::runtime_error("PFM dimensions are invalid");
    }
  }
  float scale = 0.0f;
  {
    std::istringstream scale_line(ReadHeaderLine(&input));
    if (!(scale_line >> scale) || (scale_line >> trailing) ||
        !std::isfinite(scale) || scale == 0.0f) {
      throw std::runtime_error("PFM scale is invalid");
    }
  }
  if (image.width > std::numeric_limits<size_t>::max() / image.height ||
      image.width * image.height > std::numeric_limits<size_t>::max() / 3) {
    throw std::runtime_error("PFM dimensions overflow the pixel buffer");
  }

  const size_t row_values = 3 * image.width;
  image.pixels.resize(row_values * image.height);
  const bool file_little_endian = scale < 0.0f;
  const bool host_little_endian = std::endian::native == std::endian::little;
  const float multiplier = std::abs(scale);
  std::vector<uint32_t> row(row_values);
  for (size_t reverse_y = 0; reverse_y < image.height; ++reverse_y) {
    input.read(reinterpret_cast<char *>(row.data()),
               static_cast<std::streamsize>(row.size() * sizeof(uint32_t)));
    if (input.gcount() !=
        static_cast<std::streamsize>(row.size() * sizeof(uint32_t))) {
      throw std::runtime_error("PFM pixel payload is truncated");
    }
    const size_t y = image.height - 1 - reverse_y;
    for (size_t index = 0; index < row_values; ++index) {
      uint32_t bits = row[index];
      if (file_little_endian != host_little_endian)
        bits = ByteSwap32(bits);
      const float value = std::bit_cast<float>(bits) * multiplier;
      if (!std::isfinite(value)) {
        throw std::runtime_error("PFM contains a non-finite pixel");
      }
      image.pixels[y * row_values + index] = value;
    }
  }
  if (input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("PFM has trailing data after its pixel payload");
  }
  return image;
}

template <typename Integer>
[[nodiscard]] Integer ParseInteger(std::string_view text,
                                   std::string_view option) {
  Integer value{};
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto [position, error] = std::from_chars(begin, end, value);
  if (error != std::errc{} || position != end) {
    throw std::runtime_error("Invalid integer for " + std::string(option) +
                             ": " + std::string(text));
  }
  return value;
}

[[nodiscard]] float ParseDistance(std::string_view text) {
  size_t position = 0;
  const float value = std::stof(std::string(text), &position);
  if (position != text.size() || !std::isfinite(value) || value <= 0.0f ||
      value > 25.0f) {
    throw std::runtime_error("Distance must be in (0, 25]");
  }
  return value;
}

[[nodiscard]] Options ParseCommandLine(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << "usage: gjxl_libjxl_comparison_benchmark --input IMAGE.pfm "
                   "--raw-samples OUTPUT.json [--output OUTPUT.jxl] "
                   "[--distance D] [--effort 1..10] [--num-threads N] "
                   "[--warmups N] [--samples N] [--stage-profile]\n";
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--stage-profile") {
#if GJXL_LIBJXL_STAGE_PROFILE
      options.stage_profile = true;
      continue;
#else
      throw std::runtime_error(
          "This comparison benchmark was built without stage profiling");
#endif
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("Comparison option is missing its value");
    }
    const std::string_view value = argv[++index];
    if (argument == "--input") {
      options.input = value;
    } else if (argument == "--raw-samples") {
      options.raw_samples = value;
    } else if (argument == "--output") {
      options.output = value;
    } else if (argument == "--distance") {
      options.distance = ParseDistance(value);
    } else if (argument == "--effort") {
      options.effort = ParseInteger<int>(value, argument);
      if (options.effort < 1 || options.effort > 10) {
        throw std::runtime_error("Effort must be between 1 and 10");
      }
    } else if (argument == "--num-threads") {
      options.num_threads = ParseInteger<size_t>(value, argument);
    } else if (argument == "--warmups") {
      options.warmups = ParseInteger<size_t>(value, argument);
    } else if (argument == "--samples") {
      options.samples = ParseInteger<size_t>(value, argument);
      if (options.samples == 0) {
        throw std::runtime_error("At least one measured sample is required");
      }
    } else {
      throw std::runtime_error("Unknown comparison benchmark option: " +
                               std::string(argument));
    }
  }
  if (options.input.empty() || options.raw_samples.empty()) {
    throw std::runtime_error("--input and --raw-samples are required");
  }
  return options;
}

void RequireSuccess(JxlEncoderStatus status, std::string_view operation) {
  if (status != JXL_ENC_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(static_cast<int>(status)));
  }
}

[[nodiscard]] std::vector<uint8_t>
Encode(const PfmImage &image, const Options &options, void *runner
#if GJXL_LIBJXL_STAGE_PROFILE
       ,
       jxl::EncoderStageProfileSink *stage_profile
#endif
) {
  JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
  if (!encoder)
    throw std::runtime_error("Unable to allocate libjxl encoder");
  RequireSuccess(JxlEncoderSetParallelRunner(encoder.get(),
                                             JxlThreadParallelRunner, runner),
                 "JxlEncoderSetParallelRunner");
  RequireSuccess(JxlEncoderUseContainer(encoder.get(), JXL_FALSE),
                 "JxlEncoderUseContainer");

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = static_cast<uint32_t>(image.width);
  info.ysize = static_cast<uint32_t>(image.height);
  info.bits_per_sample = 32;
  info.exponent_bits_per_sample = 8;
  info.uses_original_profile = JXL_FALSE;
  RequireSuccess(JxlEncoderSetBasicInfo(encoder.get(), &info),
                 "JxlEncoderSetBasicInfo");

  JxlColorEncoding color;
  JxlColorEncodingSetToLinearSRGB(&color, JXL_FALSE);
  RequireSuccess(JxlEncoderSetColorEncoding(encoder.get(), &color),
                 "JxlEncoderSetColorEncoding");

  JxlEncoderFrameSettings *frame =
      JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
  if (frame == nullptr) {
    throw std::runtime_error("Unable to allocate libjxl frame settings");
  }
#if GJXL_LIBJXL_STAGE_PROFILE
  RequireSuccess(
      JxlEncoderFrameSettingsSetStageProfileForBenchmark(frame, stage_profile),
      "Attach libjxl stage profile");
#endif
  RequireSuccess(JxlEncoderSetFrameDistance(frame, options.distance),
                 "JxlEncoderSetFrameDistance");
  RequireSuccess(JxlEncoderFrameSettingsSetOption(
                     frame, JXL_ENC_FRAME_SETTING_EFFORT, options.effort),
                 "Set libjxl effort");
  RequireSuccess(
      JxlEncoderFrameSettingsSetOption(frame, JXL_ENC_FRAME_SETTING_MODULAR, 0),
      "Force libjxl VarDCT mode");

  const JxlPixelFormat format{3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  RequireSuccess(JxlEncoderAddImageFrame(frame, &format, image.pixels.data(),
                                         image.pixels.size() * sizeof(float)),
                 "JxlEncoderAddImageFrame");
  JxlEncoderCloseInput(encoder.get());

  std::vector<uint8_t> output(4096);
  uint8_t *next = output.data();
  size_t available = output.size();
  JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
  while (status == JXL_ENC_NEED_MORE_OUTPUT) {
    status = JxlEncoderProcessOutput(encoder.get(), &next, &available);
    if (status == JXL_ENC_NEED_MORE_OUTPUT) {
      const size_t used = static_cast<size_t>(next - output.data());
      if (output.size() > std::numeric_limits<size_t>::max() / 2) {
        throw std::runtime_error("libjxl output size overflow");
      }
      output.resize(output.size() * 2);
      next = output.data() + used;
      available = output.size() - used;
    }
  }
  RequireSuccess(status, "JxlEncoderProcessOutput");
  output.resize(static_cast<size_t>(next - output.data()));
  return output;
}

void WriteBytes(const std::filesystem::path &destination,
                const std::vector<uint8_t> &bytes) {
  std::ofstream output(destination, std::ios::binary | std::ios::trunc);
  if (!output || !output.write(reinterpret_cast<const char *>(bytes.data()),
                               static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("Unable to write comparison codestream: " +
                             destination.string());
  }
}

void WriteRawSamples(const std::filesystem::path &destination,
                     const Options &options, const PfmImage &image,
                     const std::vector<Sample> &samples) {
  std::filesystem::path temporary = destination;
  temporary += ".tmp-" + std::to_string(static_cast<uint64_t>(
                             Clock::now().time_since_epoch().count()));
  try {
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(temporary, std::ios::out | std::ios::trunc);
    output << "{\n"
           << "  \"schema_version\": " << (options.stage_profile ? 2 : 1)
           << ",\n";
    if (options.stage_profile) {
      output << "  \"timing_semantics\": {\"elapsed_nanoseconds\": "
                "\"complete-encode-wall-time\", \"phase_nanoseconds\": "
                "\"wall-clock-barrier-time\", \"work_nanoseconds\": "
                "\"aggregate-worker-time\"},\n";
    } else {
      output << "  \"timing_semantics\": "
                "\"complete-encode-wall-time\",\n";
    }
    output << "  \"stage_profile_enabled\": "
           << (options.stage_profile ? "true" : "false") << ",\n"
           << "  \"encoder\": \"libjxl\",\n"
           << "  \"revision\": \"" << GJXL_LIBJXL_REVISION << "\",\n"
           << "  \"input_width\": " << image.width << ",\n"
           << "  \"input_height\": " << image.height << ",\n"
           << "  \"input_layout\": \"interleaved-linear-srgb-f32\",\n"
           << "  \"thread_count\": " << options.num_threads << ",\n"
           << "  \"requested_distance\": " << std::setprecision(9)
           << options.distance << ",\n"
           << "  \"effort\": " << options.effort << ",\n"
           << "  \"validation_encodes\": 1,\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"sample_count\": " << options.samples << ",\n"
           << "  \"samples\": [\n";
    for (size_t index = 0; index < samples.size(); ++index) {
      const Sample &sample = samples[index];
      output << "    {\"sample_index\": " << sample.index
             << ", \"elapsed_nanoseconds\": " << sample.elapsed_nanoseconds
             << ", \"encoded_bytes\": " << sample.encoded_bytes;
#if GJXL_LIBJXL_STAGE_PROFILE
      if (options.stage_profile) {
        output << ", \"phase_nanoseconds\": {";
        for (size_t stage = 0; stage < kPhaseNames.size(); ++stage) {
          if (stage != 0)
            output << ", ";
          output << '\"' << kPhaseNames[stage]
                 << "\": " << sample.stage_profile.phase_nanoseconds[stage];
        }
        output << "}, \"work_nanoseconds\": {";
        for (size_t stage = 0; stage < kWorkNames.size(); ++stage) {
          if (stage != 0)
            output << ", ";
          output << '\"' << kWorkNames[stage]
                 << "\": " << sample.stage_profile.work_nanoseconds[stage];
        }
        output << "}, \"invocation_counts\": {\"phase\": {";
        for (size_t stage = 0; stage < kPhaseNames.size(); ++stage) {
          if (stage != 0)
            output << ", ";
          output << '\"' << kPhaseNames[stage]
                 << "\": " << sample.stage_profile.phase_invocations[stage];
        }
        output << "}, \"work\": {";
        for (size_t stage = 0; stage < kWorkNames.size(); ++stage) {
          if (stage != 0)
            output << ", ";
          output << '\"' << kWorkNames[stage]
                 << "\": " << sample.stage_profile.work_invocations[stage];
        }
        output << "}}, \"counts\": {";
        for (size_t count = 0; count < kCountNames.size(); ++count) {
          if (count != 0)
            output << ", ";
          output << '\"' << kCountNames[count]
                 << "\": " << sample.stage_profile.counts[count];
        }
        output << "}";
      }
#endif
      output << '}';
      if (index + 1 != samples.size())
        output << ',';
      output << '\n';
    }
    output << "  ]\n}\n";
    output.close();
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
      throw std::runtime_error(
          "Could not atomically replace raw comparison output: " +
          error.message());
    }
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

#if GJXL_LIBJXL_STAGE_PROFILE
void ValidateStageProfile(const Sample &sample) {
  const auto &profile = sample.stage_profile;
  if (profile.overflowed) {
    throw std::runtime_error("libjxl stage profile counter overflowed");
  }
  uint64_t phase_sum = 0;
  for (size_t stage = 0; stage + 1 < kPhaseNames.size(); ++stage) {
    if (profile.phase_nanoseconds[stage] == 0 ||
        profile.phase_invocations[stage] == 0) {
      throw std::runtime_error("libjxl stage profile has an empty wall phase");
    }
    phase_sum += profile.phase_nanoseconds[stage];
  }
  const size_t complete =
      static_cast<size_t>(jxl::EncoderProfilePhase::kCompleteSerializer);
  if (phase_sum != profile.phase_nanoseconds[complete] ||
      profile.phase_invocations[complete] != 1) {
    throw std::runtime_error(
        "libjxl serializer phase union does not match its wall phases");
  }
  for (size_t work = 0; work < kWorkNames.size(); ++work) {
    if (profile.work_nanoseconds[work] == 0 ||
        profile.work_invocations[work] == 0) {
      throw std::runtime_error("libjxl stage profile has an empty work stage");
    }
  }
  const size_t output_bytes =
      static_cast<size_t>(jxl::EncoderProfileCount::kOutputBytes);
  if (profile.counts[output_bytes] != sample.encoded_bytes) {
    throw std::runtime_error(
        "libjxl stage profile output byte count does not match the output");
  }
}

void ValidateStableStageProfile(const Sample &expected, const Sample &actual) {
  if (expected.stage_profile.phase_invocations !=
          actual.stage_profile.phase_invocations ||
      expected.stage_profile.work_invocations !=
          actual.stage_profile.work_invocations ||
      expected.stage_profile.counts != actual.stage_profile.counts) {
    throw std::runtime_error(
        "libjxl stage profile counts changed between measured samples");
  }
}
#endif

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = ParseCommandLine(argc, argv);
    const PfmImage image = ReadPfm(options.input);
    if (image.width > std::numeric_limits<uint32_t>::max() ||
        image.height > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("PFM dimensions exceed the libjxl API limit");
    }
    JxlThreadParallelRunnerPtr runner =
        JxlThreadParallelRunnerMake(nullptr, options.num_threads);
    if (!runner) {
      throw std::runtime_error("Unable to allocate libjxl parallel runner");
    }

    std::vector<uint8_t> encoded;
    // The GJXL public-workflow benchmark performs one untimed validation
    // encode before its requested warmups. Mirror it so both measured paths
    // enter the sample loop with the same number of complete prior encodes.
    encoded = Encode(image, options, runner.get()
#if GJXL_LIBJXL_STAGE_PROFILE
                                         ,
                     nullptr
#endif
    );
    for (size_t warmup = 0; warmup < options.warmups; ++warmup) {
      encoded = Encode(image, options, runner.get()
#if GJXL_LIBJXL_STAGE_PROFILE
                                           ,
                       nullptr
#endif
      );
    }
    std::vector<Sample> samples;
    samples.reserve(options.samples);
    std::vector<uint8_t> expected;
    for (size_t index = 0; index < options.samples; ++index) {
      Sample sample;
      sample.index = index;
      const Clock::time_point begin = Clock::now();
      encoded = Encode(image, options, runner.get()
#if GJXL_LIBJXL_STAGE_PROFILE
                                           ,
                       options.stage_profile ? &sample.stage_profile : nullptr
#endif
      );
      const uint64_t elapsed = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                               begin)
              .count());
      if (index == 0) {
        expected = encoded;
      } else if (encoded != expected) {
        throw std::runtime_error(
            "libjxl codestream changed between measured samples");
      }
      sample.elapsed_nanoseconds = elapsed;
      sample.encoded_bytes = encoded.size();
#if GJXL_LIBJXL_STAGE_PROFILE
      if (options.stage_profile) {
        const size_t output_bytes =
            static_cast<size_t>(jxl::EncoderProfileCount::kOutputBytes);
        sample.stage_profile.counts[output_bytes] = encoded.size();
        ValidateStageProfile(sample);
        if (!samples.empty())
          ValidateStableStageProfile(samples.front(), sample);
      }
#endif
      samples.push_back(std::move(sample));
    }

    WriteRawSamples(options.raw_samples, options, image, samples);
    if (!options.output.empty())
      WriteBytes(options.output, encoded);
    std::cout << "libjxl comparison benchmark: " << image.width << 'x'
              << image.height << " distance=" << options.distance
              << " effort=" << options.effort
              << " threads=" << options.num_threads
              << " samples=" << options.samples
              << " encoded_bytes=" << encoded.size() << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
