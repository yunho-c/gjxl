// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Compile separately against each revision's matching headers and libraries.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "codestream/workflow_internal.h"
#include "io/pfm.h"
#include "codestream/batch_workflow.h"
#ifdef GJXL_FINAL_SCHEDULER
#include "codestream/workflow_admission.h"
#endif

#include <atomic>
#include <bit>
#include <exception>
#include <filesystem>
#include <fstream>
#include <latch>
#include <mach/mach.h>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D image_extent) : extent(image_extent) {
    size_t pixel_count = 0;
    if (extent.empty() || !extent.try_area(&pixel_count)) {
      throw std::runtime_error("Benchmark image extent is invalid");
    }
    for (std::vector<float>& values : plane) {
      values.resize(pixel_count);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
        gjxl::PlaneF32View{plane[0].data(), extent, extent.width},
        gjxl::PlaneF32View{plane[1].data(), extent, extent.width},
        gjxl::PlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
        gjxl::ConstPlaneF32View{plane[0].data(), extent, extent.width},
        gjxl::ConstPlaneF32View{plane[1].data(), extent, extent.width},
        gjxl::ConstPlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> plane;
};
[[nodiscard]] ImageStorage LoadPfm(std::string_view path) {
  gjxl::Image3FBuffer decoded;
  const gjxl::Status status = gjxl::io::ReadPfm(
    std::filesystem::path(path), &decoded);
  if (!status.ok()) {
    throw std::runtime_error(
      "Unable to open benchmark PFM: " + std::string(path) + ": " +
      std::string(status.message()));
  }
  ImageStorage image(decoded.extent());
  for (size_t channel = 0; channel < image.plane.size(); ++channel) {
    std::copy(
      decoded.plane(channel).begin(), decoded.plane(channel).end(),
      image.plane[channel].begin());
  }
  return image;
}

void FillSynthetic(ImageStorage* image) {
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      const float fx =
          static_cast<float>(x) /
          static_cast<float>(std::max<size_t>(1, image->extent.width - 1));
      const float fy =
          static_cast<float>(y) /
          static_cast<float>(std::max<size_t>(1, image->extent.height - 1));
      image->plane[0][y * image->extent.width + x] =
          std::clamp(0.08f + 0.72f * fx +
                         0.13f * std::sin(0.47f * static_cast<float>(x + y)),
                     0.0f, 1.0f);
      image->plane[1][y * image->extent.width + x] = std::clamp(
          0.1f + 0.68f * fy +
              0.16f * std::cos(0.39f * (2.0f * static_cast<float>(x) -
                                        static_cast<float>(y))),
          0.0f, 1.0f);
      image->plane[2][y * image->extent.width + x] =
          ((x / 7 + y / 5) & 1u) == 0 ? 0.12f : 0.84f;
    }
  }
}

void RequireStatus(std::string_view operation, gjxl::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(operation) + ": " + std::string(status.message()));
}
uint64_t Ns(Clock::duration d) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}
uint64_t Footprint(bool peak = false) {
  task_vm_info_data_t info{};
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &n) != KERN_SUCCESS)
    throw std::runtime_error("task_info failed");
  return peak ? info.ledger_phys_footprint_peak : info.phys_footprint;
}
void Check(bool good, const char* message) { if (!good) throw std::runtime_error(message); }
size_t Number(const std::string& text) {
  size_t used = 0;
  Check(!text.empty() && text[0] != '-', "Invalid unsigned argument");
  const auto value = std::stoull(text, &used);
  Check(used == text.size() && value <= std::numeric_limits<size_t>::max(), "Invalid unsigned argument");
  return static_cast<size_t>(value);
}
uint64_t HashBytes(std::span<const uint8_t> bytes) {
  uint64_t hash = 14695981039346656037ull;
  for (uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ull;
  return hash;
}
uint64_t HashImage(const ImageStorage& image) {
  uint64_t hash = 14695981039346656037ull;
  for (const auto& plane : image.plane) for (float value : plane) {
    const auto bits = std::bit_cast<uint32_t>(value);
    for (size_t j = 0; j < 4; ++j) hash = (hash ^ ((bits >> (8 * j)) & 255)) * 1099511628211ull;
  }
  return hash;
}
struct Call {
  std::vector<gjxl::VarDctBatchEncodingRequest> requests;
  std::vector<gjxl::VarDctBatchEncodingResult> results;
  std::vector<uint8_t> bytes;
  gjxl::VarDctEncodingSummary summary;
  gjxl::VarDctEncodingTiming timing;
  gjxl::Status status;
  std::exception_ptr exception;
  uint64_t wall_ns = 0;
};
} // namespace

int main(int argc, char** argv) try {
  size_t count = 9, batch = 0, in_flight = 1, callers = 1, threads = 0, cpu_limit = 0;
  std::string limit_mode = "default", retain;
  std::vector<ImageStorage> original;
  std::vector<std::string> names;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    Check(++i < argc, "Missing argument value");
    const std::string value = argv[i];
    if (key == "--count") count = Number(value);
    else if (key == "--batch") batch = Number(value);
    else if (key == "--in-flight") in_flight = Number(value);
    else if (key == "--callers") callers = Number(value);
    else if (key == "--threads") threads = Number(value);
    else if (key == "--cpu-limit") cpu_limit = Number(value);
    else if (key == "--limit") limit_mode = value;
    else if (key == "--retain") retain = value;
    else if (key == "--input") {
      original.push_back(LoadPfm(value)); names.push_back(value);
    } else if (key == "--synthetic") {
      const auto separator = value.find('x');
      Check(separator != std::string::npos, "Synthetic input requires WIDTHxHEIGHT");
      ImageStorage image({Number(value.substr(0, separator)), Number(value.substr(separator + 1))});
      FillSynthetic(&image); original.push_back(std::move(image)); names.push_back(value);
    } else throw std::runtime_error("Unknown argument: " + key);
  }
  Check(!original.empty() && count > 0 && in_flight > 0 && callers > 0 && callers <= 2,
        "Invalid input/count/concurrency");
  Check(batch != 0 || callers == 1, "Multiple callers require explicit batches");
  Check(batch == 0 || batch % original.size() == 0, "Mixed batch must contain equal counts of each shape");
  if (!retain.empty()) Check(std::filesystem::is_directory(retain), "Retention directory must already exist");
  auto changed = original;
  for (auto& image : changed) for (auto& plane : image.plane) for (float& value : plane) value = 0.91f * value + 0.017f;
  gjxl::VarDctEncodingOptions options;
  options.backend = gjxl::VarDctBackendPreference::kMetal;
  options.metal_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  options.effort = 7; options.butteraugli_target = 1.2f; options.cpu_thread_count = threads;
  const auto setup_begin = Clock::now();
  RequireStatus("Backend setup", gjxl::codestream_internal::EnsureProductionMetalBackendAvailable());
  const uint64_t setup_ns = Ns(Clock::now() - setup_begin);
  size_t limit = 0, planned_slots = batch ? in_flight : 1;
#ifdef GJXL_FINAL_SCHEDULER
  std::vector<gjxl::codestream_internal::WorkflowStoragePlan> plans(original.size());
  size_t maximum_single = 0;
  for (size_t i = 0; i < original.size(); ++i) {
    RequireStatus("Plan", gjxl::codestream_internal::PlanWorkflowAdmission(original[i].extent,
      {options, gjxl::codestream_internal::WorkflowStorageRoute::kCpu,
       gjxl::codestream_internal::WorkflowStorageAdapter::kBorrowedLinearRgb, true}, nullptr, false, true, &plans[i]));
    maximum_single = std::max(maximum_single, plans[i].working.peak_bytes);
  }
  gjxl::codestream_internal::BatchWorkflowStorageAccumulator accumulator;
  gjxl::codestream_internal::BatchWorkflowStoragePlan batch_plan;
  if (batch) {
    for (size_t j = 0; j < batch; ++j) RequireStatus("Accumulate", accumulator.AddRequest(&plans[j % plans.size()]));
    RequireStatus("Batch plan", accumulator.Finish(in_flight, 0, &batch_plan));
  }
  if (limit_mode == "tight") limit = batch ? batch_plan.minimum_required_bytes : maximum_single;
  else if (limit_mode == "full") {
    const size_t per_call = batch ? batch_plan.working.peak_bytes : maximum_single;
    Check(per_call <= std::numeric_limits<size_t>::max() / callers, "Combined memory limit overflow");
    limit = callers * per_call;
  } else Check(limit_mode == "default" || limit_mode == "unlimited", "Unknown memory policy");
  if (limit_mode != "default" || cpu_limit != 0)
    RequireStatus("Domain", gjxl::ExecutionDomain::Create({limit, cpu_limit}, &options.execution_domain));
  const auto domain = options.execution_domain ? options.execution_domain : gjxl::ExecutionDomain::Default();
  if (batch) {
    RequireStatus("Bound batch", accumulator.Finish(in_flight, limit, &batch_plan));
    planned_slots = batch_plan.in_flight;
  }
#else
  Check(limit_mode == "default" && cpu_limit == 0, "Integrated baseline has no shared domain API");
#endif
  std::vector<std::unique_ptr<gjxl::VarDctBatchEncoder>> drivers(callers);
  if (batch) for (auto& driver : drivers) RequireStatus("Create batch", gjxl::VarDctBatchEncoder::Create(in_flight, &driver));
  const uint64_t initial_footprint = Footprint();
  std::cout << "{\"type\":\"setup\",\"setup_ns\":" << setup_ns << ",\"batch\":" << batch
            << ",\"in_flight\":" << in_flight << ",\"planned_slots\":" << planned_slots << ",\"callers\":" << callers
            << ",\"threads\":" << threads << ",\"cpu_limit_requested\":" << cpu_limit << ",\"managed_limit\":" << limit
            << ",\"initial_footprint\":" << initial_footprint << ",\"sources\":[";
  for (size_t i = 0; i < original.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << "{\"name\":" << std::quoted(names[i]) << ",\"width\":" << original[i].extent.width
              << ",\"height\":" << original[i].extent.height << ",\"original_fnv64\":\"" << HashImage(original[i])
              << "\",\"changed_fnv64\":\"" << HashImage(changed[i]) << "\"}";
  }
  std::cout << "]}\n" << std::flush;
  for (size_t iteration = 0; iteration < count; ++iteration) {
    std::vector<Call> calls(callers);
    for (size_t c = 0; c < callers; ++c) for (size_t j = 0; j < batch; ++j) {
      const size_t source = (j + c) % original.size();
      const bool phase = (iteration + j + c) % 2;
      calls[c].requests.push_back({(phase ? changed[source] : original[source]).ConstView(), options});
    }
    const size_t single_source = iteration % original.size();
    const bool single_phase = (iteration / original.size()) % 2;
    const auto run = [&](size_t c) {
      try {
        const auto begin = Clock::now();
        calls[c].status = batch ? drivers[c]->Encode(calls[c].requests, &calls[c].results) :
          gjxl::EncodeLinearRgbVarDctCodestreamProfiled(
            (single_phase ? changed[single_source] : original[single_source]).ConstView(), options,
            &calls[c].bytes, &calls[c].summary, &calls[c].timing);
        calls[c].wall_ns = Ns(Clock::now() - begin);
      } catch (...) { calls[c].exception = std::current_exception(); }
    };
    uint64_t cohort_ns = 0;
    if (callers == 1) { run(0); cohort_ns = calls[0].wall_ns; }
    else {
      std::latch ready(callers), start(1);
      std::atomic<bool> cancelled{false};
      std::vector<std::thread> workers;
      try {
        for (size_t c = 0; c < callers; ++c) workers.emplace_back([&, c] {
          ready.count_down(); start.wait(); if (!cancelled) run(c);
        });
      } catch (...) {
        cancelled = true; start.count_down();
        for (auto& worker : workers) worker.join();
        throw;
      }
      ready.wait();
      const auto begin = Clock::now();
      start.count_down();
      for (auto& worker : workers) worker.join();
      cohort_ns = Ns(Clock::now() - begin);
    }
    for (const auto& call : calls) {
      if (call.exception) std::rethrow_exception(call.exception);
      RequireStatus("Public call", call.status);
      if (batch) Check(call.results.size() == batch, "Batch changed result count");
      for (const auto& result : call.results) RequireStatus("Image", result.status);
    }
    std::cout << "{\"type\":\"sample\",\"index\":" << iteration << ",\"cohort_ns\":" << cohort_ns
              << ",\"footprint_with_outputs\":" << Footprint() << ",\"peak_footprint\":" << Footprint(true)
              << ",\"managed\":";
#ifdef GJXL_FINAL_SCHEDULER
    const auto s = domain->snapshot();
    Check(s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 && s.suspended_cpu_workers == 0 &&
          s.waiting_cpu_callers == 0 && s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit &&
          s.active_reservations == 0 && s.waiting_requests == 0 && (limit == 0 || s.peak_committed_bytes <= limit),
          "Shared CPU/memory invariant failed");
    std::cout << "{\"peak_backing\":" << s.peak_backing_bytes << ",\"peak_committed\":" << s.peak_committed_bytes
              << ",\"idle\":" << s.idle_capacity_bytes << ",\"live\":" << s.live_capacity_bytes
              << ",\"cpu_peak\":" << s.peak_cpu_participants << ",\"cpu_protected\":" << s.peak_cpu_protected_slots
              << ",\"cpu_limit\":" << s.effective_cpu_participant_limit << '}';
#else
    std::cout << "null"; // Baseline predates both public admission and complete managed accounting.
#endif
    std::cout << ",\"calls\":[";
    for (size_t c = 0; c < callers; ++c) {
      if (c) std::cout << ',';
      std::cout << "{\"caller\":" << c << ",\"wall_ns\":" << calls[c].wall_ns << ",\"images\":[";
      for (size_t j = 0; j < (batch ? batch : 1); ++j) {
        if (j) std::cout << ',';
        const size_t source = batch ? (j + c) % original.size() : single_source;
        const bool phase = batch ? (iteration + j + c) % 2 : single_phase;
        const auto& bytes = batch ? calls[c].results[j].codestream : calls[c].bytes;
        std::cout << "{\"source\":" << source << ",\"phase\":" << int(phase) << ",\"bytes\":" << bytes.size()
                  << ",\"fnv64\":\"" << HashBytes(bytes) << "\",\"scheduling\":";
#ifdef GJXL_FINAL_SCHEDULER
        if (batch) {
          const auto& t = calls[c].results[j].scheduling;
          Check(t.cpu_admitted && t.queue_nanoseconds + t.service_nanoseconds == t.ready_nanoseconds &&
                t.ready_nanoseconds <= calls[c].wall_ns, "Image timing exceeds complete public call");
          std::cout << "{\"queue_ns\":" << t.queue_nanoseconds << ",\"service_ns\":" << t.service_nanoseconds
                    << ",\"ready_ns\":" << t.ready_nanoseconds << '}';
        } else std::cout << "null";
#else
        std::cout << "null";
#endif
        std::cout << '}';
        if (!retain.empty()) {
          const auto file = std::filesystem::path(retain) / ("sample-" + std::to_string(iteration) + "-caller-" +
            std::to_string(c) + "-image-" + std::to_string(j) + ".jxl");
          Check(!std::filesystem::exists(file), "Refusing to overwrite retained output");
          std::ofstream stream(file, std::ios::binary);
          stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
          Check(bool(stream), "Unable to retain codestream");
        }
      }
      std::cout << "]}";
    }
    std::cout << "]}\n" << std::flush;
  }
  const uint64_t idle_footprint = Footprint(); // Outputs released, backend and drivers still alive.
#ifdef GJXL_FINAL_SCHEDULER
  for (auto& driver : drivers) if (driver) driver->Shutdown();
  RequireStatus("Trim", gjxl::codestream_internal::WorkflowAdmission::TrimIdle(*domain));
  const auto final = domain->snapshot();
  Check(final.live_capacity_bytes == 0 && final.idle_capacity_bytes == 0 && final.reserved_unbacked_bytes == 0 &&
        final.active_reservations == 0 && final.waiting_requests == 0 && final.active_cpu_participants == 0 &&
        final.reserved_cpu_workers == 0 && final.suspended_cpu_workers == 0 && final.waiting_cpu_callers == 0,
        "Post-trim shared domain is not empty");
#else
  drivers.clear();
  RequireStatus("Trim", gjxl::TrimVarDctPreparationCache());
#endif
  std::cout << "{\"type\":\"trim\",\"backend_alive_idle_footprint\":" << idle_footprint
            << ",\"post_trim_footprint\":" << Footprint() << ",\"peak_footprint\":" << Footprint(true) << "}\n";
  return EXIT_SUCCESS;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return EXIT_FAILURE; }
