// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_sample_preparation.h"
#include "codec/modular/histogram_tree_parallel.h"
#include "codestream/workflow_admission_scope.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/modular_resident_histogram_fitting.h"
#include "io/pnm.h"
#include "frontier_sample_archive.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::histogram_gpu_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void Check(bool b, const char *m) { if (!b) throw std::runtime_error(m); }
void Ok(Status s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
struct Observation {
  double wall = 0, cpu = 0, restore = 0, gpu_call = 0;
  uint64_t gpu_ns = 0;
  size_t submissions = 0, rounds = 0, ready = 0, active = 0, same_fit = 0;
  size_t histogram = 0, deferred = 0;
  size_t input_backing = 0, peak_backing = 0, incremental_peak = 0, nodes = 0;
};
Observation Run(GpuBackend &gpu, std::span<const ResidentHistogramRequest> requests,
                std::span<const DecisionTree> expected, unsigned mode,
                const ResidentHistogramInput &controls) {
  // 0: CPU 1; 1: CPU 14; 2/3: DFS/frontier GPU all; 4/5: half-frontier/half-CPU,
  // sequential or concurrent, with 13 CPU workers plus one GPU caller.
  const size_t gpu_count = mode < 2 ? 0 : mode < 4 ? requests.size() : requests.size() / 2;
  const size_t cpu_count = requests.size() - gpu_count;
  const size_t participants = mode == 0 || mode == 2 || mode == 3 ? 1 : mode == 1 ? 14 : 13;
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{2} << 30, 14}, &domain));
  const auto &budget = codestream_internal::WorkflowAdmission::Budget(*domain);
  ResourceReservation reservation;
  Ok(budget.TryReserve(size_t{2} << 30, &reservation));
  auto context = CurrentResourceContext(); context.domain = domain.get(); context.reservation = &reservation;
  ResourceContextScope scope(context);
  CpuExecutionScope execution; Ok(execution.Start(domain, 14));
  EncodeScope threads(14);
  Observation result;
  {
    std::vector<HistogramTrainingSamples> copies;
    copies.reserve(cpu_count);
    const auto restore = Clock::now();
    for (size_t i = gpu_count; i < requests.size(); ++i) copies.push_back(*requests[i].source.samples);
    result.restore = Seconds(restore);
    std::vector<DecisionTree> cpu_trees(cpu_count);
    std::vector<Status> statuses(cpu_count);
    ResidentHistogramResult resident;
    histogram_parallel_internal::PropertyTeam team(participants);
    Ok(team.Start()); Check(team.participants() == participants, "CPU benchmark grant differs");
    result.input_backing = budget.snapshot().total.live_capacity_bytes;
    const auto cpu_work = [&] {
      team.Run(cpu_count, [&](size_t i, size_t) noexcept {
        statuses[i] = LearnHistogramTree(copies[i], &cpu_trees[i], requests[gpu_count + i].learning);
      });
    };
    const auto gpu_work = [&] {
      if (!gpu_count) return;
      const auto start = Clock::now();
      ResidentHistogramInput input = controls;
      input.requests = requests.first(gpu_count);
      input.executor = mode == 2 ? ResidentHistogramExecutor::kDepthFirst : controls.executor;
      Ok(QueryGpuModularHistogramFitting(gpu)->FitHistogramTrees(input, &resident));
      result.gpu_call = Seconds(start);
    };
    const auto start = Clock::now(); const auto cpu_start = std::clock();
    if (mode == 5) {
      std::exception_ptr failure;
      std::thread pending([&] {
        try {
          ResourceContextScope worker_context(context);
          CpuExecutionScope cpu; Ok(cpu.Start(domain, 1));
          gpu_work();
        } catch (...) { failure = std::current_exception(); }
      });
      cpu_work();
      { CpuSuspension suspension; pending.join(); }
      if (failure) std::rethrow_exception(failure);
    } else { gpu_work(); cpu_work(); }
    result.wall = Seconds(start); result.cpu = double(std::clock() - cpu_start) / CLOCKS_PER_SEC;
    for (const auto &s : statuses) Ok(s);
    for (size_t i = 0; i < requests.size(); ++i) {
      const auto &tree = i < gpu_count ? resident.trees[i] : cpu_trees[i - gpu_count];
      Check(tree == expected[i], "Complete fitting tree differs from native CPU");
      result.nodes += tree.size;
    }
    result.gpu_ns = resident.gpu_nanoseconds;
    result.submissions = resident.submissions; result.rounds = resident.frontier_rounds;
    result.ready = resident.maximum_ready_nodes;
    result.active = resident.maximum_active_nodes; result.same_fit = resident.maximum_same_fit_nodes;
    result.histogram = resident.maximum_histogram_bytes; result.deferred = resident.maximum_deferred_nodes;
    result.peak_backing = budget.snapshot().peak_backing_bytes;
    result.incremental_peak = result.peak_backing - result.input_backing;
  }
  reservation.Reset();
  Check(!budget.snapshot().committed_bytes(), "Fitting probe leaked managed backing");
  Check(domain->snapshot().peak_cpu_protected_slots <= 14, "Fitting probe exceeded CPU allowance");
  return result;
}
}
int main(int argc, char **argv) try {
  Check(argc >= 3 && argc <= 10, "resident_fitting_probe INPUT.pam|SAMPLES.gjfs OUTPUT.json [fits=16 [repeats=6 [tile=128 [cpu-frontier|cpu-tiled [active_nodes=128 [rounds=16 [scratch_mib=64]]]]]]]");
  Check(!std::filesystem::exists(argv[2]), "Refusing to replace probe output");
  size_t fits = argc > 3 ? std::stoul(argv[3]) : 16;
  const size_t repeats = argc > 4 ? std::stoul(argv[4]) : 6;
  const size_t tile = argc > 5 ? std::stoul(argv[5]) : 128;
  const bool cpu_frontier_only = argc >= 7;
  ResidentHistogramInput controls{};
  controls.maximum_active_nodes = argc > 7 ? std::stoul(argv[7]) : 128;
  controls.rounds_per_submission = argc > 8 ? std::stoul(argv[8]) : 16;
  const size_t scratch_mib = argc > 9 ? std::stoul(argv[9]) : 64;
  Check(controls.maximum_active_nodes > 0 && controls.maximum_active_nodes <= 1024 &&
        controls.rounds_per_submission > 0 && controls.rounds_per_submission <= 64 &&
        scratch_mib > 0 && scratch_mib <= 512, "Invalid frontier probe controls");
  controls.maximum_scratch_bytes = scratch_mib * 1024 * 1024;
  const bool from_archive = std::filesystem::path(argv[1]).extension() == ".gjfs";
  Check(!cpu_frontier_only || std::string_view(argv[6]) == "cpu-frontier" || std::string_view(argv[6]) == "cpu-tiled", "Unknown replay modes");
  const bool tiled = cpu_frontier_only && std::string_view(argv[6]) == "cpu-tiled";
  controls.executor = tiled ? ResidentHistogramExecutor::kTiledFrontier : ResidentHistogramExecutor::kFrontier;
  Check(fits >= 2 && fits <= 64 && repeats > 0 && repeats <= 30 && tile > 0 && tile <= 256,
        "Invalid fitting probe dimensions");
  std::vector<PreparedHistogramSamples> prepared;
  frontier_diagnostic::Archive archive;
  std::vector<ResidentHistogramRequest> requests;
  std::vector<DecisionTree> expected;
  std::vector<std::array<size_t, 3>> locations;
  requests.reserve(fits);
  std::unique_ptr<GpuBackend> gpu;
  const char *metallib = std::getenv("GJXL_FRONTIER_PROBE_METALLIB");
  if (metallib && *metallib) Ok(CreateMetalBackend(metallib, &gpu));
  else Ok(CreateEmbeddedMetalBackend({}, &gpu));
  size_t rows = 0, population = 0;
  if (from_archive) {
    archive = frontier_diagnostic::Load(argv[1]);
    fits = archive.samples.size();
    for (size_t i = 0; i < fits; ++i)
      requests.push_back({{archive.ids[i], &archive.samples[i]}, archive.options[i]});
  } else {
    io::IntegerImage image; Ok(io::ReadPnm(argv[1], &image));
    Check(image.format == ModularPixelFormat::kRgb8 && image.extent.width >= tile && image.extent.height >= tile,
          "Probe requires RGB8 input at least one tile large");
    prepared.resize(fits);
    for (size_t i = 0; i < fits; ++i) {
      const size_t x0 = (i * 977) % (image.extent.width - tile + 1);
      const size_t y0 = (i * 613) % (image.extent.height - tile + 1), channel = i % 3;
      std::vector<int32_t> pixels(tile * tile);
      for (size_t y = 0; y < tile; ++y) for (size_t x = 0; x < tile; ++x)
        pixels[y * tile + x] = image.bytes[(y0 + y) * image.row_stride + (x0 + x) * 3 + channel];
      const Extent2D extent{tile, tile};
      const ModularChannelView plane{pixels, {extent}, tile};
      const HistogramTrainingImage training{extent, {&plane, 1}};
      Ok(PrepareHistogramSamples({&training, 1}, &prepared[i]));
      requests.push_back({{static_cast<uint32_t>(i), &prepared[i].samples()}, prepared[i].tree_options()});
      locations.push_back({x0, y0, channel});
    }
  }
  expected.resize(fits);
  for (size_t i = 0; i < fits; ++i) {
    auto copy = *requests[i].source.samples;
    Ok(LearnHistogramTree(copy, &expected[i], requests[i].learning));
    rows += copy.weights.size(); population += copy.total_samples;
  }
  ResidentHistogramInput input = controls;
  input.requests = requests;
  const auto limits = QueryGpuModularHistogramFitting(*gpu)->histogram_fitting_limits();
  ResidentHistogramPlan plan; Ok(ComputeResidentHistogramPlan(input, &plan, &limits));
  std::ofstream out(argv[2]); out << std::setprecision(17) << std::boolalpha;
  out << "{\"schema\":5,\"input\":" << std::quoted(argv[1]) << ",\"fits\":" << fits
      << ",\"metallib\":" << std::quoted(metallib ? metallib : "embedded")
      << ",\"source_kind\":" << std::quoted(from_archive ? "prepared_archive" : "photo_tiles")
      << ",\"tile\":" << (from_archive ? 0 : tile) << ",\"rows\":" << rows << ",\"population\":" << population
      << ",\"planned_gpu_executor\":" << std::quoted(tiled ? "tiled_frontier" : "frontier")
      << ",\"requested_active_nodes\":" << controls.maximum_active_nodes
      << ",\"rounds_per_submission\":" << controls.rounds_per_submission
      << ",\"maximum_scratch_bytes\":" << controls.maximum_scratch_bytes
      << ",\"device_maximum_scratch_bytes\":" << limits.maximum_scratch_bytes
      << ",\"planned_active_nodes\":" << plan.frontier.active_nodes
      << ",\"planned_scratch_bytes\":" << plan.scratch_bytes
      << ",\"scratch_slot_bytes\":" << plan.frontier.slot.bytes
      << ",\"histogram_pool_bytes\":" << plan.tiled.histogram_bytes
      << ",\"planned_gpu_work_bytes\":" << plan.work_bytes
      << ",\"planned_gpu_input_bytes\":" << plan.prepared.storage_bytes
      << ",\"planned_gpu_peak_bytes\":" << plan.working.peak_bytes
      << ",\"locations\":[";
  for (size_t i = 0; i < locations.size(); ++i) out << (i ? "," : "") << '[' << locations[i][0] << ',' << locations[i][1] << ',' << locations[i][2] << ']';
  const char *names[] = {"cpu1", "cpu14", "resident_depth_first", "frontier", "split_sequential", "split_concurrent"};
  if (tiled) names[3] = "tiled_frontier";
  out << "],\"boundary\":\"Complete fitting including validation, scratch, partition, node/queue management, canonicalization and publication. GPU upload included. Input sampling, CPU input restoration, backend/domain and worker-pool creation excluded. Concurrent thread launch/join included. CPU seconds are process CPU time.\",\"observations\":[";
  bool comma = false;
  for (size_t iteration = 0; iteration <= repeats; ++iteration) {
    for (unsigned position = 0; position < (cpu_frontier_only ? 2 : 6); ++position) {
      const unsigned mode = cpu_frontier_only ? ((position + iteration) % 2 ? 3 : 1) : (position + iteration) % 6;
      const auto r = Run(*gpu, requests, expected, mode, controls);
      out << (comma ? "," : "") << "{\"mode\":" << std::quoted(names[mode]) << ",\"iteration\":" << iteration
          << ",\"warmup\":" << (iteration == 0) << ",\"exact\":true,\"wall_seconds\":" << r.wall
          << ",\"process_cpu_seconds\":" << r.cpu << ",\"input_restore_seconds\":" << r.restore
          << ",\"gpu_call_seconds\":" << r.gpu_call << ",\"gpu_nanoseconds\":" << r.gpu_ns
          << ",\"gpu_submissions\":" << r.submissions << ",\"frontier_rounds\":" << r.rounds
          << ",\"maximum_ready_nodes\":" << r.ready
          << ",\"maximum_active_nodes\":" << r.active << ",\"maximum_same_fit_nodes\":" << r.same_fit
          << ",\"maximum_histogram_bytes\":" << r.histogram << ",\"maximum_deferred_nodes\":" << r.deferred
          << ",\"tree_nodes\":" << r.nodes << ",\"input_and_pool_backing_bytes\":" << r.input_backing
          << ",\"peak_backing_bytes\":" << r.peak_backing << ",\"incremental_peak_bytes\":" << r.incremental_peak << '}';
      comma = true; out.flush();
      std::cout << iteration << ' ' << names[mode] << ' ' << r.wall << " s exact\n" << std::flush;
    }
  }
  out << "],\"passed\":true}\n"; Check(bool(out), "Probe write failed");
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
