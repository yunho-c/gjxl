// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Fixed synthetic node batches: isolate submission controls without a changing
// CPU fitting schedule. Exact native CPU candidates are the result oracle.
#include "gpu/metal/metal_backend.h"
#include "modular_histogram_fitting_test_helpers.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace histogram_fitting_test;
namespace {
using Clock = std::chrono::steady_clock;
uint64_t Nanoseconds(Clock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
}
template <class A> void Array(std::ostream &out, const A &a) {
  out << '[';
  for (size_t i = 0; i < a.size(); ++i) out << (i ? "," : "") << a[i];
  out << ']';
}
void CapacityProbe(std::ostream &out, GpuBackend &gpu, size_t repetitions) {
  auto &op = *QueryGpuModularHistogramFitting(gpu);
  struct Batch {
    size_t fits, rows;
    std::vector<HistogramTrainingSamples> samples;
    std::vector<HistogramTreeFitter> fitters;
    std::unique_ptr<PreparedGpuHistogramRows> prepared;
    std::vector<HistogramFittingNodeInput> nodes;
    std::vector<HistogramPropertyResult> expected;
    HistogramFittingEvaluationPlan plan;
  };
  std::array<Batch, 3> batches{{{1, 16384}, {10, 32768}, {32, 131072}}};
  for (auto &b : batches) {
    b.samples.reserve(b.fits); b.fitters.resize(b.fits); b.expected.resize(b.fits * 16);
    for (size_t i = 0; i < b.fits; ++i) b.samples.push_back(Fixture(6 + 3 * i, b.rows, 2));
    std::vector<HistogramFittingSource> sources;
    std::vector<HistogramFittingNodeShape> shapes;
    for (size_t i = 0; i < b.fits; ++i) {
      sources.push_back({uint32_t(i + 1), &b.samples[i]});
      Ok(b.fitters[i].Initialize(b.samples[i])); Ok(b.fitters[i].Advance());
      Check(b.fitters[i].ready(), "Variable probe root is not ready");
      b.nodes.push_back(HistogramFittingNodeView(i, b.fitters[i].node()));
      shapes.push_back(b.nodes.back().shape);
      Ok(EvaluateHistogramNodeOnCpu(b.fitters[i], sources[i].source_id,
                                   std::span(b.expected).subspan(i * 16, 16)));
    }
    Ok(op.PrepareHistogramRows(sources, &b.prepared));
    Ok(ComputeHistogramFittingEvaluationPlan(b.prepared->shapes(), shapes, {}, &b.plan));
  }
  ResourceBudget budget(256 * 1024 * 1024);
  constexpr size_t orders[6][3] = {{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};
  constexpr size_t sequence[] = {2, 0, 1, 0, 2};
  out << "{\"schema\":2,\"mode\":\"capacity\",\"synthetic\":true,\"warmups\":3,\"repetitions\":"
      << repetitions << ",\"records\":[";
  bool comma = false;
  for (size_t iteration = 0; iteration < repetitions + 3; ++iteration) {
    for (size_t variant : orders[iteration % 6]) {
      Ok(gpu.TrimPreparationCache()); // Each policy starts the same size sequence cold.
      HistogramFittingExecutionOptions options;
      options.reuse_buffers = variant != 0;
      options.reuse_larger_buffers = variant == 2;
      for (size_t step = 0; step < std::size(sequence); ++step) {
        auto &b = batches[sequence[step]];
        const auto capacities = op.HistogramFittingBufferCapacities(b.plan, options);
        HostStorageBound bound;
        Ok(ComputeHistogramFittingCapacityBound(b.plan, capacities, &bound));
        ResourceReservation reservation;
        Ok(budget.TryReserve(bound.peak_bytes, &reservation));
        ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
        HistogramFittingBatchProfile p;
        std::unique_ptr<PreparedGpuHistogramEvaluation> result;
        const auto start = Clock::now();
        Ok(op.EvaluateHistogramNodes({b.prepared.get(), b.nodes, options,
            HistogramFittingProfileMode::kWall, &p, true, capacities}, &result));
        const auto call_ns = Nanoseconds(start);
        Check(SameResults(result->results(), b.expected), "Variable CPU/GPU results differ");
        const auto release = Clock::now(); result.reset();
        const auto release_ns = Nanoseconds(release);
        reservation.Reset();
        const auto &t = p.wait_trace;
        out << (comma ? "," : "") << "{\"fits\":" << b.fits << ",\"rows_per_fit\":" << b.rows
            << ",\"variant\":" << variant << ",\"iteration\":" << iteration << ",\"step\":" << step
            << ",\"warmup\":" << (iteration < 3 ? "true" : "false")
            << ",\"exact\":true,\"call_ns\":" << call_ns << ",\"output_release_ns\":" << release_ns
            << ",\"reused_buffers\":" << p.reused_buffers
            << ",\"reused_buffer_bytes\":" << p.reused_buffer_bytes
            << ",\"compute_encoders\":" << p.compute_encoders
            << ",\"idle_bytes\":" << budget.snapshot().total.idle_capacity_bytes
            << ",\"buffer_requested_bytes\":"; Array(out, p.buffer_requested_bytes);
        out << ",\"buffer_capacity_bytes\":"; Array(out, p.buffer_capacity_bytes);
        out << ",\"wall_ns\":"; Array(out, p.wall_nanoseconds);
        out << ",\"wait_trace\":{\"commit_begin\":" << t.commit_begin
            << ",\"kernel_begin\":" << t.kernel_begin << ",\"kernel_end\":" << t.kernel_end
            << ",\"gpu_begin\":" << t.gpu_begin << ",\"gpu_end\":" << t.gpu_end
            << ",\"blocking_end\":" << t.blocking_end << "}}";
        comma = true;
      }
    }
  }
  Ok(gpu.TrimPreparationCache());
  Check(!budget.snapshot().committed_bytes() && !budget.snapshot().open_reservations,
        "Variable probe leaked storage");
  out << "],\"passed\":true}\n";
}
}
int main(int argc, char **argv) try {
  Check(argc >= 2 && argc <= 4, "histogram_submission_probe OUTPUT.json [repetitions [fixed|capacity]]");
  Check(!std::filesystem::exists(argv[1]), "Refusing to overwrite probe output");
  const size_t repetitions = argc >= 3 ? std::stoul(argv[2]) : 20;
  const std::string mode = argc == 4 ? argv[3] : "fixed";
  Check(mode == "fixed" || mode == "capacity", "Invalid probe mode");
  Check(repetitions > 0 && repetitions <= 1000, "Invalid repetition count");
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  auto *op = QueryGpuModularHistogramFitting(*gpu);
  Check(op, "No native Metal histogram fitter");
  std::ofstream out(argv[1]);
  Check(bool(out), "Cannot create probe output");
  if (mode == "capacity") {
    CapacityProbe(out, *gpu, repetitions);
    Check(bool(out), "Capacity probe output write failed");
    return 0;
  }
  out << "{\"schema\":1,\"synthetic\":true,\"warmups\":3,\"repetitions\":"
      << repetitions << ",\"records\":[";
  bool comma = false;
  for (const auto [fits, rows] : {std::pair{1u, 16384u}, {10u, 32768u}, {32u, 131072u}}) {
    Ok(gpu->TrimPreparationCache());
    std::vector<HistogramTrainingSamples> samples;
    samples.reserve(fits);
    for (size_t i = 0; i < fits; ++i) samples.push_back(Fixture(6 + 3 * i, rows, 2));
    std::vector<HistogramFittingSource> sources;
    for (size_t i = 0; i < fits; ++i) sources.push_back({uint32_t(i + 1), &samples[i]});
    std::unique_ptr<PreparedGpuHistogramRows> prepared;
    Ok(op->PrepareHistogramRows(sources, &prepared));
    std::vector<HistogramTreeFitter> fitters(fits);
    std::vector<HistogramFittingNodeInput> nodes;
    std::vector<HistogramPropertyResult> expected(fits * 16);
    for (size_t i = 0; i < fits; ++i) {
      Ok(fitters[i].Initialize(samples[i]));
      Ok(fitters[i].Advance());
      Check(fitters[i].ready(), "Synthetic root does not require evaluation");
      nodes.push_back(HistogramFittingNodeView(i, fitters[i].node()));
      Ok(EvaluateHistogramNodeOnCpu(fitters[i], sources[i].source_id,
                                   std::span(expected).subspan(i * 16, 16)));
    }
    constexpr size_t orders[4][4] = {
        {0, 1, 3, 2}, {1, 2, 0, 3}, {2, 3, 1, 0}, {3, 0, 2, 1}};
    for (size_t iteration = 0; iteration < repetitions + 3; ++iteration) {
      // Balance positions and adjacent pairs over four rounds. Reuse cases
      // share only idle exact-capacity arenas; fresh cases never consume them.
      for (size_t order = 0; order < 4; ++order) {
        const size_t variant = orders[iteration % 4][order];
        HistogramFittingExecutionOptions options;
        options.reuse_buffers = variant & 1;
        options.reuse_larger_buffers = false; // Preserve the exact-size controls.
        options.multiple_encoders = variant & 2;
        HistogramFittingBatchProfile profile;
        std::unique_ptr<PreparedGpuHistogramEvaluation> result;
        const auto start = Clock::now();
        Ok(op->EvaluateHistogramNodes({prepared.get(), nodes, options,
            HistogramFittingProfileMode::kWall, &profile, true}, &result));
        const auto call_ns = Nanoseconds(start);
        Check(SameResults(result->results(), expected), "Fixed-batch CPU/GPU result differs");
        const auto release_start = Clock::now();
        result.reset();
        const auto release_ns = Nanoseconds(release_start);
        const auto &t = profile.wait_trace;
        out << (comma ? "," : "") << "{\"fits\":" << fits << ",\"rows_per_fit\":" << rows
            << ",\"variant\":" << variant << ",\"iteration\":" << iteration
            << ",\"warmup\":" << (iteration < 3 ? "true" : "false")
            << ",\"exact\":true,\"call_ns\":" << call_ns << ",\"output_release_ns\":" << release_ns
            << ",\"nodes\":" << profile.nodes << ",\"rows\":" << profile.rows
            << ",\"tasks\":" << profile.tasks << ",\"waves\":" << profile.waves
            << ",\"scratch_bytes\":" << profile.scratch_bytes
            << ",\"reused_buffers\":" << profile.reused_buffers
            << ",\"reused_buffer_bytes\":" << profile.reused_buffer_bytes
            << ",\"compute_encoders\":" << profile.compute_encoders
            << ",\"wall_ns\":";
        Array(out, profile.wall_nanoseconds);
        out << ",\"wait_trace\":{\"commit_begin\":" << t.commit_begin
            << ",\"commit_end\":" << t.commit_end << ",\"kernel_begin\":" << t.kernel_begin
            << ",\"kernel_end\":" << t.kernel_end << ",\"gpu_begin\":" << t.gpu_begin
            << ",\"gpu_end\":" << t.gpu_end << ",\"wait_begin\":" << t.wait_begin
            << ",\"blocking_begin\":" << t.blocking_begin << ",\"blocking_end\":" << t.blocking_end
            << ",\"cpu_resumed\":" << t.cpu_resumed << "}}";
        comma = true;
      }
    }
    std::cout << fits << " fixed nodes x " << rows << " rows: exact\n";
  }
  Ok(gpu->TrimPreparationCache());
  out << "],\"passed\":true}\n";
  Check(bool(out), "Probe output write failed");
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
