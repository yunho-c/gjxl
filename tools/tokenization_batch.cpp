// SPDX-License-Identifier: Apache-2.0
// Use the existing benchmark's image loader and complete-call validation
// helper.
#define main gjxl_original_image_batch_main
#include "../benchmarks/image_batch_benchmark.cpp"
#undef main
#include "codestream/workflow_storage_plan.h"
int main(int argc, char **argv) {
  try {
    if (argc != 7 && argc != 8)
      throw std::runtime_error("input.pfm effort cpu_participants batch "
                               "samples new_output_directory [memory_bytes]");
    const std::filesystem::path input = argv[1], out = argv[6];
    const int effort = std::stoi(argv[2]);
    const size_t cpu = std::stoul(argv[3]), batch = std::stoul(argv[4]),
                 samples = std::stoul(argv[5]);
    if (!std::filesystem::create_directory(out))
      throw std::runtime_error("Output directory must be new");
    const auto image = LoadImage(input);
    std::shared_ptr<const gjxl::ExecutionDomain> domain;
    const size_t memory_bytes = argc == 8 ? std::stoull(argv[7]) : 0;
    auto status = gjxl::ExecutionDomain::Create({memory_bytes, cpu}, &domain);
    if (!status.ok())
      throw std::runtime_error(std::string(status.message()));
    gjxl::VarDctEncodingOptions options;
    options.effort = effort;
    options.butteraugli_target = 1.9f;
    options.backend = gjxl::VarDctBackendPreference::kMetal;
    options.gpu_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
    options.cpu_thread_count = cpu;
    options.execution_domain = domain;
    std::vector<uint8_t> expected;
    gjxl::VarDctEncodingSummary summary;
    const auto start = Clock::now();
    status = gjxl::EncodeLinearRgbVarDctCodestream(image.View(), options,
                                                   &expected, &summary);
    const auto cold_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             Clock::now() - start)
                             .count();
    if (!status.ok())
      throw std::runtime_error(std::string(status.message()));
    {
      std::ofstream file(out / "reference.jxl", std::ios::binary);
      file.write(reinterpret_cast<const char *>(expected.data()),
                 expected.size());
    }
    std::unique_ptr<gjxl::VarDctBatchEncoder> driver;
    status = gjxl::VarDctBatchEncoder::Create(batch, &driver);
    if (!status.ok())
      throw std::runtime_error(std::string(status.message()));
    std::vector<gjxl::VarDctBatchEncodingRequest> requests(
        batch, {image.View(), options});
    gjxl::codestream_internal::WorkflowStoragePlan image_plan;
    status = gjxl::codestream_internal::ComputeWorkflowStoragePlan(
        image.View().extent(),
        {options, gjxl::codestream_internal::WorkflowStorageRoute::kMetal,
         gjxl::codestream_internal::WorkflowStorageAdapter::kBorrowedLinearRgb,
         true}, &image_plan);
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    gjxl::codestream_internal::BatchWorkflowStorageAccumulator accumulator;
    for (size_t i = 0; i < batch; ++i) {
      status = accumulator.AddRequest(&image_plan);
      if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    }
    gjxl::codestream_internal::BatchWorkflowStoragePlan plan;
    status = accumulator.Finish(batch, memory_bytes, &plan);
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    {
      std::ofstream file(out / "plan.json");
      file << "{\"memory_limit\":" << memory_bytes
           << ",\"in_flight\":" << plan.in_flight
           << ",\"work_slot_bytes\":" << plan.work_slot_bytes
           << ",\"minimum_required_bytes\":" << plan.minimum_required_bytes
           << ",\"trim_after_each_image\":" << plan.trim_after_each_image
           << "}\n";
    }
    std::ofstream csv(out / "samples.csv");
    csv << "sample,batch,wall_ns,encoded_bytes_each,peak_cpu_slots,peak_"
           "backing_bytes,peak_committed_bytes,idle_bytes\n";
    csv << "-2,1," << cold_ns << ',' << expected.size() << ",0,0,0,0\n";
    for (int sample = -1; sample < int(samples); ++sample) {
      const auto elapsed = RunBatch(*driver, requests, expected, summary);
      const auto snapshot = domain->snapshot();
      if (snapshot.peak_cpu_protected_slots > cpu ||
          snapshot.active_cpu_participants || snapshot.reserved_cpu_workers ||
          snapshot.active_reservations)
        throw std::runtime_error("Batch exceeded CPU cap or leaked admission");
      csv << sample << ',' << batch << ',' << elapsed << ',' << expected.size()
          << ',' << snapshot.peak_cpu_protected_slots << ','
          << snapshot.peak_backing_bytes << ',' << snapshot.peak_committed_bytes
          << ',' << snapshot.idle_capacity_bytes << '\n';
      csv.flush();
      std::cout << sample << " B" << batch << " ms=" << elapsed / 1e6
                << " peak_backing=" << snapshot.peak_backing_bytes << '\n'
                << std::flush;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
