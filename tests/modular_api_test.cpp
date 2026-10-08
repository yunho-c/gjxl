// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gjxl/modular.hpp"
#include "gjxl/gjxl.h"
#include "gjxl/execution_domain.hpp"
#include "codestream/modular/search.h"
#include "codestream/modular/workflow.h"
#include "core/worker_launch_internal.h"
#include "modular_reference.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <barrier>
#include "codestream/workflow.h"
#include <thread>
#include <stdexcept>
namespace {
using namespace gjxl;
namespace m = gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
void Check(bool b, const char *text) {
  if (!b)
    throw std::runtime_error(text);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
size_t cases = 0;
void Case(ModularPixelFormat format, bool big, Extent2D extent, ModularEntropy entropy,
          bool search) {
  m::ModularInputProfile profile;
  Ok(m::ResolveModularInput(extent, static_cast<m::PackedModularFormat>(format), &profile));
  const size_t row = extent.width * profile.channel_count * profile.bytes_per_sample,
               stride = row + 3;
  std::vector<uint8_t> bytes(1 + stride * (extent.height - 1) + row, 0xcd);
  ref::IntegerImage expected{extent.width,
                             extent.height,
                             static_cast<uint32_t>(profile.channel_count),
                             profile.metadata.bits,
                             {}};
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < profile.channel_count; ++c) {
        const uint16_t sample =
            static_cast<uint16_t>((c == 3 && x % 3 == 0 ? 0 : ((x + y) % 16) * 313 + c * 27) &
                                  (profile.metadata.bits == 8 ? 255 : 65535));
        expected.samples.push_back(sample);
        const size_t i =
            1 + y * stride + (x * profile.channel_count + c) * profile.bytes_per_sample;
        bytes[i] =
            static_cast<uint8_t>(big && profile.bytes_per_sample == 2 ? sample >> 8 : sample);
        if (profile.bytes_per_sample == 2)
          bytes[i + 1] = static_cast<uint8_t>(big ? sample : sample >> 8);
      }
  ModularImageView input{std::span(bytes).subspan(1), extent, stride, format,
                         big ? ModularByteOrder::kBigEndian : ModularByteOrder::kLittleEndian};
  ModularEncodingOptions options;
  options.cpu_thread_count = 1;
  options.entropy = entropy;
  options.search = search;
  std::vector<uint8_t> serial, parallel;
  ModularEncodingSummary summary;
  Ok(EncodeModularImage(input, options, &serial, &summary));
  Check(summary.extent == extent && summary.encoded_bytes == serial.size() &&
            summary.total_seconds > 0 && summary.backend == ModularBackend::kCpu,
        "Wrong summary");
  m::ModularWorkflowStoragePlan plan;
  auto mode =
      entropy == ModularEntropy::kPrefix ? EntropyCodingMode::kPrefix : EntropyCodingMode::kAns;
  Ok(search ? m::ComputeModularSearchStoragePlan(
                  extent, static_cast<m::PackedModularFormat>(format), mode, &plan, 4)
            : m::ComputeModularWorkflowStoragePlan(
                  extent, static_cast<m::PackedModularFormat>(format), mode, {}, &plan, 4));
  Ok(ExecutionDomain::Create(
      {.managed_memory_bytes = plan.working.peak_bytes + plan.output.peak_bytes,
       .cpu_participant_limit = 4},
      &options.execution_domain));
  options.cpu_thread_count = 4;
  Ok(EncodeModularImage(input, options, &parallel));
  Check(serial == parallel && ref::DecodeLossless(parallel) == expected,
        "Parallel/native decode mismatch");
  auto snapshot = options.execution_domain->snapshot();
  Check(snapshot.peak_cpu_protected_slots <= 4 && snapshot.active_cpu_participants == 0 &&
            snapshot.active_reservations == 0,
        "Leaked CPU or reservation");
  if (extent.width > 256)
    Check(snapshot.peak_cpu_protected_slots > 1, "Parallel path did not execute");
  GJXLContextOptions context_options;
  Check(gjxl_context_options_init(&context_options, sizeof(context_options)) == GJXL_OK,
        "Context init failed");
  context_options.num_cpu_threads = 4;
  GJXLContext *context = nullptr;
  Check(gjxl_context_create(&context_options, &context) == GJXL_OK, "Context failed");
  GJXLPixelFormat cformat = profile.channel_count == 1   ? GJXL_PIXEL_FORMAT_GRAY8_SRGB
                            : profile.channel_count == 3 ? GJXL_PIXEL_FORMAT_RGB8_SRGB
                                                         : GJXL_PIXEL_FORMAT_RGBA8_SRGB;
  if (profile.bytes_per_sample == 2)
    cformat = static_cast<GJXLPixelFormat>((profile.channel_count == 1   ? 4
                                            : profile.channel_count == 3 ? 6
                                                                         : 8) +
                                           big);
  GJXLImageView view{sizeof(GJXLImageView),
                     static_cast<uint32_t>(extent.width),
                     static_cast<uint32_t>(extent.height),
                     cformat,
                     bytes.data() + 1,
                     bytes.size() - 1,
                     stride};
  GJXLModularOptions coptions;
  Check(gjxl_modular_options_init(&coptions, sizeof(coptions)) == GJXL_OK, "Modular init failed");
  coptions.search = search;
  coptions.entropy = static_cast<GJXLModularEntropy>(entropy);
  GJXLBuffer output{};
  Check(gjxl_encode_modular(context, &view, &coptions, &output) == GJXL_OK, gjxl_get_last_error());
  Check(std::ranges::equal(serial, std::span(output.data, output.size)), "C ABI bytes mismatch");
  gjxl_buffer_free(&output);
  gjxl_context_destroy(context);
  for (auto backend : {ModularBackend::kMetal, ModularBackend::kCuda}) {
    options.backend = backend;
    Check(EncodeModularImage(input, options, &parallel, &summary).code() ==
                  StatusCode::kUnsupported &&
              parallel == serial,
          "Forced GPU silently fell back or changed output");
  }
  ++cases;
}
void LaunchFailures() {
  using namespace thread_budget_internal;
  std::vector<uint8_t> pixels(769 * 17 * 3, 19);
  ModularImageView image{pixels, {769, 17}, 769 * 3};
  ModularEncodingOptions options;
  options.cpu_thread_count = 4;
  Ok(ExecutionDomain::Create({.cpu_participant_limit = 4}, &options.execution_domain));
  for (auto site : {WorkerLaunchSite::kModularTokenization, WorkerLaunchSite::kModularEmission})
    for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
      for (size_t before : {size_t{0}, size_t{1}}) {
        WorkerLaunchFaultForTesting fault{site, before, kind};
        WorkerLaunchFaultScopeForTesting scope(&fault);
        std::vector<uint8_t> output{7, 8, 9};
        auto status = EncodeModularImage(image, options, &output);
        Check(!status.ok() && fault.triggered && fault.launched_in_group == before &&
                  output == std::vector<uint8_t>({7, 8, 9}),
              "Launch failure not atomic");
        const auto snapshot = options.execution_domain->snapshot();
        Check(snapshot.active_cpu_participants == 0 && snapshot.reserved_cpu_workers == 0 &&
                  snapshot.suspended_cpu_workers == 0 && snapshot.active_reservations == 0,
              "Launch failure leaked capacity");
        Ok(EncodeModularImage(image, options, &output));
      }
}
void MixedDomain() {
  constexpr size_t limit = 512 * 1024 * 1024;
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({.managed_memory_bytes = limit, .cpu_participant_limit = 2}, &domain));
  std::array<Status, 4> statuses;
  std::barrier start(4);
  std::array<std::thread, 4> workers;
  for (size_t i = 0; i < 4; ++i)
    workers[i] = std::thread([&, i] {
      start.arrive_and_wait();
      std::vector<uint8_t> output;
      if (i % 2) {
        std::array<float, 3> pixel{0.1f, 0.3f, 0.7f};
        ConstImage3FView image{{ConstPlaneF32View{&pixel[0], {1, 1}, 1},
                                ConstPlaneF32View{&pixel[1], {1, 1}, 1},
                                ConstPlaneF32View{&pixel[2], {1, 1}, 1}}};
        VarDctEncodingOptions options;
        options.backend = VarDctBackendPreference::kCpu;
        options.cpu_thread_count = 2;
        options.execution_domain = domain;
        options.effort = 1;
        statuses[i] = EncodeLinearRgbVarDctCodestream(image, options, &output);
      } else {
        std::vector<uint8_t> pixels(513 * 17 * 4, 71);
        ModularEncodingOptions options;
        options.execution_domain = domain;
        options.cpu_thread_count = 4;
        options.search = true;
        statuses[i] = EncodeModularImage({pixels, {513, 17}, 513 * 4, ModularPixelFormat::kRgba8},
                                         options, &output);
      }
    });
  for (auto &worker : workers)
    worker.join();
  for (auto status : statuses)
    Ok(status);
  const auto snapshot = domain->snapshot();
  Check(snapshot.peak_backing_bytes <= limit && snapshot.peak_cpu_protected_slots <= 2 &&
            snapshot.active_cpu_participants == 0 && snapshot.active_reservations == 0,
        "Mixed domain cap or drain failed");
}
void PublicationFailures() {
  using namespace resource_budget_internal;
  GJXLContextOptions context_options;
  Check(gjxl_context_options_init(&context_options, sizeof(context_options)) == GJXL_OK,
        "Context init failed");
  context_options.num_cpu_threads = 1;
  GJXLContext *context = nullptr;
  Check(gjxl_context_create(&context_options, &context) == GJXL_OK, "Context create failed");
  std::array<uint8_t, 8 * 9 * 4> pixels{};
  GJXLImageView image{sizeof(GJXLImageView), 8,    9, GJXL_PIXEL_FORMAT_RGBA8_SRGB, pixels.data(),
                      pixels.size(),         8 * 4};
  size_t count = 0;
  for (; count < 4000; ++count) {
    GJXLBuffer output{};
    ArmManagedHostAllocationFailureAfterForTest(count);
    auto result = gjxl_encode_modular(context, &image, nullptr, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (result == GJXL_OK) {
      Check(pending, "Missed C failure position");
      gjxl_buffer_free(&output);
      break;
    }
    Check(result == GJXL_ERROR_OUT_OF_MEMORY && output.data == nullptr && output.size == 0 &&
              !pending,
          "C publication failure was not atomic");
  }
  Check(count > 10 && count < 4000, "C publication sweep incomplete");
  gjxl_context_destroy(context);
  std::cout << count << " C publication failure positions passed\n";
}
void Prefixes() {
  struct Prefix {
    uint32_t size;
    uint32_t canary;
  } options{4, 0x12345678};
  Check(gjxl_modular_options_init(reinterpret_cast<GJXLModularOptions *>(&options), 4) == GJXL_OK &&
            options.canary == 0x12345678,
        "Sized prefix overwrite");
  GJXLContext *context = nullptr;
  Check(gjxl_context_create(nullptr, &context) == GJXL_OK, "Default context failed");
  uint8_t pixel = 73;
  GJXLImageView image{sizeof(GJXLImageView), 1, 1, GJXL_PIXEL_FORMAT_GRAY8_SRGB, &pixel, 1, 1};
  GJXLBuffer output{};
  Check(gjxl_encode_modular(context, &image, reinterpret_cast<GJXLModularOptions *>(&options),
                            &output) == GJXL_OK,
        "Sized prefix encode failed");
  gjxl_buffer_free(&output);
  image.pixels = nullptr;
  Check(gjxl_encode_modular(context, &image, nullptr, &output) == GJXL_ERROR_INVALID_ARGUMENT &&
        output.data == nullptr && output.size == 0, "Null raster accepted");
  gjxl_context_destroy(context);
}
} // namespace
int main(int argc, char **argv) try {
  if (argc == 8 && std::string(argv[1]) == "--check-file") {
    std::ifstream encoded(argv[2], std::ios::binary), raw(argv[3], std::ios::binary);
    std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(encoded), {}};
    auto decoded = ref::DecodeLossless(bytes);
    Check(decoded.width == std::stoull(argv[4]) && decoded.height == std::stoull(argv[5]) &&
              decoded.channels == std::stoul(argv[6]) && decoded.bits == std::stoul(argv[7]),
          "CLI metadata mismatch");
    for (auto sample : decoded.samples) {
      const int lo = raw.get(), hi = raw.get();
      Check(lo >= 0 && hi >= 0 && sample == (lo | (hi << 8)), "CLI sample mismatch");
    }
    Check(raw.get() == EOF, "CLI sample count mismatch");
    return 0;
  }
  Prefixes();
  LaunchFailures();
  MixedDomain();
  PublicationFailures();
  for (auto format :
       {ModularPixelFormat::kGray8, ModularPixelFormat::kRgb8, ModularPixelFormat::kRgba8,
        ModularPixelFormat::kGray16, ModularPixelFormat::kRgb16, ModularPixelFormat::kRgba16})
    for (bool big : {false, true}) {
      if (big && static_cast<int>(format) < 3)
        continue;
      for (auto extent : {Extent2D{17, 19}, Extent2D{513, 17}})
        for (auto entropy : {ModularEntropy::kPrefix, ModularEntropy::kAns})
          for (bool search : {false, true})
            Case(format, big, extent, entropy, search);
    }
  std::cout << cases << " public C/C++ serial-parallel exact cases passed\n";
} catch (const std::exception &e) {
  std::cerr << "case " << cases << ": " << e.what() << '\n';
  return 1;
}
