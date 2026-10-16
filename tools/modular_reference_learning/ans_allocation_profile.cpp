// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Public complete-call timing. Decoder validation and file I/O are outside it.
#include "codec/modular/profile.h"
#include "codec/modular/frame.h"
#include "codestream/workflow_internal.h"
#include "codestream/workflow_admission_scope.h"
#include "io/pnm.h"
#include "modular_reference.h"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
namespace reference = gjxl::test::modular_reference;
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
} // namespace

int main(int argc, char** argv) try {
  Check(argc == 5, "ans_allocation_profile INPUT OUT cpu|metal PARTICIPANTS");
  const std::filesystem::path directory = argv[2];
  Check(!std::filesystem::exists(directory), "Output directory already exists");
  const std::string route = argv[3];
  const size_t participants = std::stoull(argv[4]);
  Check((route == "cpu" || route == "metal") && participants > 0 && participants <= 256,
        "Invalid route or participant count");
  io::IntegerImage input;
  Ok(io::ReadPnm(argv[1], &input));
  modular_internal::ModularInputProfile format;
  Ok(modular_internal::ResolveModularInput(input.extent,
      static_cast<modular_internal::PackedModularFormat>(input.format), &format));
  reference::IntegerImage expected{input.extent.width, input.extent.height,
    static_cast<uint32_t>(format.channel_count), format.metadata.bits, {}};
  for (size_t y = 0; y < input.extent.height; ++y)
    for (size_t x = 0; x < input.extent.width * format.channel_count; ++x) {
      const size_t p = y * input.row_stride + x * format.bytes_per_sample;
      expected.samples.push_back(format.bytes_per_sample == 1 ? input.bytes[p]
        : (uint16_t(input.bytes[p]) << 8) | input.bytes[p + 1]);
    }
  // Match the integrated profile: backend setup is outside the encoder interval.
  if (route == "metal") {
    GpuBackend* backend = nullptr;
    Ok(codestream_internal::BorrowProductionMetalBackend(&backend));
  }
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{8} << 30, participants}, &domain));
  ModularEncodingOptions options;
  options.backend = route == "cpu" ? ModularBackend::kCpu : ModularBackend::kMetal;
  options.cpu_thread_count = participants;
  options.execution_domain = domain;
  modular_internal::ModularProfile profile;
  std::vector<uint8_t> encoded;
  double wall, cpu;
  {
    modular_internal::ProfileSession session(&profile);
    const auto start = std::chrono::steady_clock::now();
    const auto cpu_start = std::clock();
    Ok(EncodeModularImage(input.view(), options, &encoded));
    wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    cpu = double(std::clock() - cpu_start) / CLOCKS_PER_SEC;
  }
  Check(reference::DecodeLossless(encoded) == expected, "Independent decode differs");
  const auto memory = codestream_internal::WorkflowAdmission::Budget(*domain).snapshot();
  Check(!memory.committed_bytes() && !memory.total.backing_count &&
        !memory.total.pending_count && !memory.open_reservations,
        "Public encode leaked managed storage");
  std::filesystem::create_directories(directory);
  std::ofstream output(directory / "output.jxl", std::ios::binary);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
  std::ofstream result(directory / "result.json");
  result.exceptions(std::ios::failbit | std::ios::badbit);
  result << std::setprecision(17) << "{\"schema\":1,\"route\":" << std::quoted(route)
         << ",\"participants\":" << participants << ",\"exact_decode\":true,\"bytes\":"
         << encoded.size() << ",\"wall\":" << wall << ",\"cpu\":" << cpu
         << ",\"stages\":[";
  for (size_t i = 0; i < profile.seconds.size(); ++i)
    result << (i ? "," : "") << profile.seconds[i];
  result << "],\"stream_jobs\":" << profile.parallel_entropy.stream_jobs
         << ",\"entropy_participants\":" << profile.parallel_entropy.peak_participants
         << ",\"model_work\":" << profile.parallel_entropy.model_work_seconds
         << ",\"emission_work\":" << profile.parallel_entropy.emission_work_seconds
         << ",\"peak_committed\":" << memory.peak_committed_bytes
         << ",\"peak_backing\":" << memory.peak_backing_bytes << "}\n";
  std::cout << route << ": " << wall << " s, " << encoded.size() << " bytes, exact decode\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
