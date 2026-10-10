// Local public-workflow boundary measurement harness.
#include "core/boundary_trace_internal.h"
#include "codestream/workflow.h"
#include "codestream/batch_workflow.h"
#include "codestream/workflow_storage_plan.h"
#include "io/pfm.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace bt = gjxl::boundary_trace_internal;
void Ok(gjxl::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
int main(int argc, char** argv) try {
  if (argc != 12) throw std::runtime_error(
      "input effort distance cpu batch samples warmups output mode memory_bytes order_seed");
  const std::filesystem::path input = argv[1], out = argv[8];
  const int effort = std::stoi(argv[2]);
  const float distance = std::stof(argv[3]);
  const size_t cpu = std::stoul(argv[4]), batch = std::stoul(argv[5]);
  const int samples = std::stoi(argv[6]), warmups = std::stoi(argv[7]);
  const std::string mode = argv[9];
  const int order_seed = std::stoi(argv[11]);
  const size_t memory = std::stoull(argv[10]);
  if (!cpu || !batch || samples < 1 || warmups < 0 ||
      (mode != "ordinary" && mode != "paired" && mode != "trace" && mode != "plan"))
    throw std::runtime_error("Invalid capture options");
#ifndef BOUNDARY_INSTRUMENTED
  if (mode != "ordinary" && mode != "plan") throw std::runtime_error("Baseline cannot trace");
#endif
  const char* tokens = std::getenv("GJXL_GPU_TOKENIZATION");
  if (!tokens || (std::string(tokens) != "0" && std::string(tokens) != "1"))
    throw std::runtime_error("Explicit GPU tokenization policy required");
  if (!std::filesystem::create_directory(out)) throw std::runtime_error("Output must be new");
  gjxl::Image3FBuffer image;
  Ok(gjxl::io::ReadPfm(input, &image));
  std::shared_ptr<const gjxl::ExecutionDomain> domain;
  Ok(gjxl::ExecutionDomain::Create({memory, cpu}, &domain));
  gjxl::VarDctEncodingOptions options;
  options.backend = gjxl::VarDctBackendPreference::kMetal;
  options.gpu_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  options.effort = effort;
  options.butteraugli_target = distance;
  options.cpu_thread_count = cpu;
  options.execution_domain = domain;
  std::unique_ptr<gjxl::VarDctBatchEncoder> driver;
  std::vector<gjxl::VarDctBatchEncodingRequest> requests(batch, {image.const_view(), options});
  if (batch > 1) Ok(gjxl::VarDctBatchEncoder::Create(batch, &driver));
  gjxl::codestream_internal::WorkflowStoragePlan image_plan;
  Ok(gjxl::codestream_internal::ComputeWorkflowStoragePlan(image.extent(),
      {options, gjxl::codestream_internal::WorkflowStorageRoute::kMetal,
       gjxl::codestream_internal::WorkflowStorageAdapter::kBorrowedLinearRgb, batch > 1}, &image_plan));
  gjxl::codestream_internal::BatchWorkflowStoragePlan plan;
  if (batch > 1) {
    gjxl::codestream_internal::BatchWorkflowStorageAccumulator accumulator;
    for (size_t i = 0; i < batch; ++i) Ok(accumulator.AddRequest(&image_plan));
    Ok(accumulator.Finish(batch, memory, &plan));
  } else {
    plan.in_flight = 1;
    plan.work_slot_bytes = image_plan.working.peak_bytes;
    plan.minimum_required_bytes = image_plan.working.peak_bytes;
  }
  {
    std::ofstream f(out / "config.json");
    f << "{\"schema\":1,\"clock\":\"mach_absolute_time_ns\",\"effort\":" << effort
      << ",\"distance\":" << distance << ",\"cpu\":" << cpu << ",\"batch\":" << batch
      << ",\"width\":" << image.extent().width << ",\"height\":" << image.extent().height
      << ",\"memory_limit\":" << memory << ",\"planned_in_flight\":" << plan.in_flight
      << ",\"work_slot_bytes\":" << plan.work_slot_bytes
      << ",\"minimum_required_bytes\":" << plan.minimum_required_bytes
      << ",\"trim_after_each_image\":" << (plan.trim_after_each_image ? "true" : "false")
      << ",\"gpu_tokenization\":" << tokens << "}\n";
  }
  if (mode == "plan") return 0;
  bt::Sink sink;
  (void)bt::Now(); (void)bt::Thread();
  std::vector<uint8_t> expected;
  gjxl::VarDctEncodingSummary expected_summary;
  std::ofstream rows(out / "samples.jsonl");
  rows.exceptions(std::ios::badbit | std::ios::failbit);
  const auto run = [&](int sample, bool traced, bool reference, int variant) {
    const auto variant_string = std::to_string(variant);
    setenv("GJXL_EXPERIMENT_EAGER_ENTROPY", variant_string.c_str(), 1);
    sink.Reset();
    std::vector<uint8_t> bytes;
    gjxl::VarDctEncodingSummary summary;
    std::vector<gjxl::VarDctBatchEncodingResult> results;
    const bool use_batch = batch > 1 && !reference;
    if (traced) bt::active_sink.store(&sink);
    const auto begin = bt::Now();
    const auto status = use_batch
      ? driver->Encode(requests, &results)
      : gjxl::EncodeLinearRgbVarDctCodestream(image.const_view(), options, &bytes, &summary);
    const auto end = bt::Now();
    bt::active_sink.store(nullptr);
    Ok(status);
    if (reference) { expected = bytes; expected_summary = summary; }
    else if (!use_batch && (bytes != expected || summary != expected_summary))
      throw std::runtime_error("Single encode bytes or summary differ");
    if (use_batch) {
      if (results.size() != batch) throw std::runtime_error("Batch result count differs");
      for (const auto& r : results) {
        Ok(r.status);
        if (r.codestream != expected || r.summary != expected_summary)
          throw std::runtime_error("Batch bytes or summary differ");
      }
    }
    if (sink.count.load() >= bt::Sink::capacity ||
        (traced && sink.jobs.load() != (use_batch ? batch : 1)))
      throw std::runtime_error("Missing trace job or trace capacity overflow");
    for (size_t i = 0; i < sink.count.load(); ++i) {
      const auto& e = sink.events[i];
      if (e.end < e.begin || e.begin < begin || e.end > end || e.job == 0)
        throw std::runtime_error(std::string("Trace outside public call: ") + e.name);
    }
    const auto snap = domain->snapshot();
    if (snap.peak_cpu_protected_slots > cpu || snap.active_reservations ||
        snap.active_cpu_participants || snap.reserved_cpu_workers)
      throw std::runtime_error("CPU cap exceeded or participation/reservation leaked");
    rusage usage{}; getrusage(RUSAGE_SELF, &usage);
    rows << "{\"sample\":" << sample << ",\"traced\":" << (traced ? "true" : "false")
         << ",\"variant\":" << variant
         << ",\"reference\":" << (reference ? "true" : "false")
         << ",\"begin_ns\":" << begin << ",\"end_ns\":" << end
         << ",\"wall_ns\":" << end - begin << ",\"batch\":" << (use_batch ? batch : 1)
         << ",\"encoded_bytes_each\":" << expected.size()
         << ",\"byte_equal\":true,\"summary_equal\":true,\"peak_cpu_slots\":" << snap.peak_cpu_protected_slots
         << ",\"peak_backing_bytes\":" << snap.peak_backing_bytes
         << ",\"peak_committed_bytes\":" << snap.peak_committed_bytes
         << ",\"idle_bytes\":" << snap.idle_capacity_bytes
         << ",\"maxrss_bytes\":" << usage.ru_maxrss << ",\"pageins\":" << usage.ru_majflt
         << ",\"scheduling\":[";
    for (size_t i = 0; i < results.size(); ++i) {
      if (i) rows << ',';
      const auto& s = results[i].scheduling;
      rows << "{\"queue_ns\":" << s.queue_nanoseconds << ",\"service_ns\":" << s.service_nanoseconds
           << ",\"ready_ns\":" << s.ready_nanoseconds << '}';
    }
    rows << "],\"events\":"; sink.Write(rows); rows << "}\n"; rows.flush();
    std::cout << variant << ' ' << sample << ' ' << (traced ? "trace" : "ordinary") << ' ' << (end - begin) / 1e6 << " ms\n" << std::flush;
  };
  run(-100, mode == "trace", true, 0);
  const int orders[6][3] = {{0,1,2},{2,1,0},{1,0,2},{2,0,1},{0,2,1},{1,2,0}};
  for (int sample = -warmups; sample < samples; ++sample) {
    const int rotation = (sample + warmups + order_seed) % 6;
    for (int j = 0; j < 3; ++j)
      run(sample, mode == "trace", false, orders[rotation][j]);
  }
  std::ofstream f(out / "reference.jxl", std::ios::binary);
  f.write(reinterpret_cast<const char*>(expected.data()), expected.size());
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
