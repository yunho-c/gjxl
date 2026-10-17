// SPDX-License-Identifier: Apache-2.0
// Compiled only by GJXL_BUILD_MODULAR_FRONTIER_DIAGNOSTICS. Not benchmark timing.
#pragma once
#include "frontier_sample_archive.h"
#include <atomic>
#include <cstdlib>
#include <sstream>
#include <set>

namespace gjxl::frontier_diagnostic {
class Capture {
public:
  Capture(const histogram_gpu_internal::ResidentHistogramInput &input, MTL::Device *device,
          MTL::ComputePipelineState *pipeline,
          const histogram_gpu_internal::ResidentHistogramPlan &plan)
      : device_(device), requests_(input.requests) {
    const char *directory = std::getenv("GJXL_FRONTIER_DIAGNOSTIC_DIR");
    if (!directory || !*directory) return;
    static std::atomic<size_t> next{0};
    cohort_ = next.fetch_add(1);
    const char *selected = std::getenv("GJXL_FRONTIER_DIAGNOSTIC_COHORT");
    if (cohort_ != (selected ? std::stoul(selected) : 0)) return;
    directory_ = directory;
    std::filesystem::create_directories(directory_);
    if (std::getenv("GJXL_FRONTIER_DUMP_ROWS")) Save(input, directory_ / "samples.gjfs");
    const char *submissions = std::getenv("GJXL_FRONTIER_CAPTURE_SUBMISSIONS");
    if (submissions && *submissions) {
      std::stringstream list(submissions); std::string item;
      while (std::getline(list, item, ',')) {
        if (selected_.size() == 8) throw std::runtime_error("At most eight capture submissions");
        selected_.insert(std::stoul(item));
      }
    }
    const auto metadata = directory_ / "submissions.jsonl";
    if (std::filesystem::exists(metadata)) throw std::runtime_error("Diagnostic output already exists");
    log_.open(metadata); log_.exceptions(std::ios::badbit | std::ios::failbit);
    if (std::getenv("GJXL_FRONTIER_PROFILE_PHASES")) {
      const auto path = directory_ / "dispatches.jsonl";
      if (std::filesystem::exists(path)) throw std::runtime_error("Dispatch diagnostic already exists");
      dispatches_.open(path); dispatches_.exceptions(std::ios::badbit | std::ios::failbit);
      // A one-round command exposes that round's final slot summaries without
      // adding a diagnostic GPU kernel or reading partially completed work.
      if (input.rounds_per_submission == 1 &&
          input.executor == histogram_gpu_internal::ResidentHistogramExecutor::kTiledFrontier) {
        const auto path = directory_ / "workload.jsonl";
        if (std::filesystem::exists(path)) throw std::runtime_error("Workload diagnostic already exists");
        workload_.open(path); workload_.exceptions(std::ios::badbit | std::ios::failbit);
      }
    }
    const auto path = directory_ / "pipeline.json";
    if (std::filesystem::exists(path)) throw std::runtime_error("Pipeline diagnostic already exists");
    std::ofstream properties(path); properties.exceptions(std::ios::badbit | std::ios::failbit);
    properties << "{\"schema\":1,\"static_threadgroup_memory_bytes\":" << pipeline->staticThreadgroupMemoryLength()
               << ",\"max_threads_per_threadgroup\":" << pipeline->maxTotalThreadsPerThreadgroup()
               << ",\"thread_execution_width\":" << pipeline->threadExecutionWidth()
               << ",\"device_max_threadgroup_memory_bytes\":" << device_->maxThreadgroupMemoryLength()
               << ",\"active_capacity\":" << plan.frontier.active_nodes
               << ",\"scratch_bytes\":" << plan.scratch_bytes << "}\n";
  }
  ~Capture() { End(); }
  bool ProfilePhases() const { return dispatches_.is_open(); }
  void RecordProfile(size_t submission, const gpu_profile_internal::GpuSubmissionProfile &profile) {
    if (!dispatches_.is_open()) return;
    size_t index = 0;
    for (const auto &stage : profile.stages) {
      if (!stage.timestamp_valid || stage.end_timestamp < stage.begin_timestamp || stage.dispatches.size() != 1)
        throw std::runtime_error("Invalid frontier phase timestamp");
      const auto &d = stage.dispatches.front();
      // Indirect arguments are reused each round. Their completed values do
      // not describe earlier dispatches, so deliberately omit grid metadata.
      dispatches_ << "{\"submission\":" << submission << ",\"dispatch\":" << index++
                  << ",\"kernel\":" << std::quoted(std::string(d.kernel_id))
                  << ",\"begin_ns\":" << stage.begin_timestamp
                  << ",\"end_ns\":" << stage.end_timestamp
                  << ",\"gpu_ns\":" << stage.gpu_nanoseconds << "}\n";
    }
    dispatches_.flush();
  }
  void Begin(size_t submission) {
    if (!selected_.contains(submission)) return;
    auto *manager = MTL::CaptureManager::sharedCaptureManager();
    if (manager->isCapturing()) throw std::runtime_error("Another Metal capture is already active");
    const auto path = directory_ / ("submission-" + std::to_string(submission) + ".gputrace");
    if (std::filesystem::exists(path)) throw std::runtime_error("GPU trace already exists");
    auto descriptor = NS::TransferPtr(MTL::CaptureDescriptor::alloc()->init());
    descriptor->setCaptureObject(device_);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(NS::URL::fileURLWithPath(NS::String::string(path.c_str(), NS::UTF8StringEncoding)));
    NS::Error *error = nullptr;
    if (!manager->startCapture(descriptor.get(), &error))
      throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "GPU capture failed");
    active_ = true;
  }
  void End() noexcept {
    if (active_) { MTL::CaptureManager::sharedCaptureManager()->stopCapture(); active_ = false; }
  }
  void Record(size_t submission, uint64_t gpu_ns, const gjxl_histogram_resident::FrontierControl &c,
              const histogram_gpu_internal::ResidentHistogramPlan &plan,
              std::span<const std::byte> completed) {
    if (!log_.is_open()) return;
    log_ << "{\"cohort\":" << cohort_ << ",\"submission\":" << submission
         << ",\"gpu_ns\":" << gpu_ns << ",\"head\":" << c.head << ",\"tail\":" << c.tail
         << ",\"rounds\":" << c.rounds << ",\"last_count\":" << c.count
         << ",\"maximum_ready\":" << c.maximum_ready << "}\n";
    log_.flush();
    const bool tiled = plan.tiled.frontier.active_nodes != 0;
    if (tiled && workload_.is_open()) {
      using namespace gjxl_histogram_resident;
      const auto &t = *reinterpret_cast<const TiledControl *>(completed.data() + plan.frontier.control);
      if (t.selected > plan.frontier.active_nodes || t.row_tasks > plan.frontier.active_nodes)
        throw std::runtime_error("Invalid diagnostic workload counts");
      workload_ << "{\"submission\":" << submission << ",\"round\":" << c.rounds
                << ",\"new_nodes\":" << t.row_tasks << ",\"pending_after\":" << t.live
                << ",\"jobs\":" << t.jobs << ",\"stats_groups\":" << t.stats_dispatch[0]
                << ",\"collect_groups\":" << t.collect_dispatch[0] << ",\"nodes\":[";
      const auto *selected = reinterpret_cast<const Word *>(completed.data() + plan.tiled.selected);
      for (size_t i = 0; i < t.selected; ++i) {
        if (selected[i] >= plan.frontier.active_nodes) throw std::runtime_error("Invalid diagnostic slot");
        const auto *base = completed.data() + plan.frontier.scratch + selected[i] * plan.frontier.slot.bytes;
        const auto &slot = *reinterpret_cast<const TiledSlot *>(base);
        if (slot.ref.fit >= plan.fits || slot.ref.node >= plan.layout[slot.ref.fit].maximum_nodes)
          throw std::runtime_error("Invalid diagnostic node reference");
        const auto &node = reinterpret_cast<const Node *>(completed.data() + plan.layout[slot.ref.fit].nodes)[slot.ref.node];
        const auto &source = *requests_[slot.ref.fit].source.samples;
        const auto *properties = reinterpret_cast<const TiledProperty *>(base + plan.tiled.properties);
        size_t local_properties = 0;
        for (size_t p = 0; p < source.property_count; ++p)
          local_properties += properties[p].bins && properties[p].histogram == kTiledLocalHistogram;
        workload_ << (i ? "," : "") << "{\"fit\":" << slot.ref.fit << ",\"node\":" << slot.ref.node
                  << ",\"rows\":" << node.end - node.begin << ",\"symbols\":" << slot.symbols
                  << ",\"histogram_bytes\":" << slot.needed << ",\"predictors\":" << source.predictor_count
                  << ",\"local_properties\":" << local_properties
                  << ",\"split\":" << (node.property >= 0 ? "true" : "false") << ",\"property_bins\":[";
        for (size_t p = 0; p < source.property_count; ++p)
          workload_ << (p ? "," : "") << properties[p].bins;
        workload_ << "]}";
      }
      workload_ << "]}\n"; workload_.flush();
    }
    const bool idle = !tiled || !reinterpret_cast<const gjxl_histogram_resident::TiledControl *>(
        completed.data() + plan.frontier.control)->live;
    if (!c.status && c.head == c.tail && idle && std::getenv("GJXL_FRONTIER_DUMP_NODES")) {
      const auto path = directory_ / "nodes.jsonl";
      if (std::filesystem::exists(path)) throw std::runtime_error("Node diagnostic already exists");
      std::ofstream nodes(path); nodes.exceptions(std::ios::badbit | std::ios::failbit);
      nodes << "{\"schema\":1,\"fits\":" << plan.fits
            << ",\"active_capacity\":" << plan.frontier.active_nodes
            << ",\"scratch_bytes\":" << plan.scratch_bytes
            << ",\"rounds\":" << c.rounds << ",\"nodes\":" << c.tail << "}\n";
      const auto *queue = reinterpret_cast<const gjxl_histogram_resident::NodeRef *>(
          completed.data() + plan.frontier.queue);
      if (c.tail > plan.frontier.queue_capacity) throw std::runtime_error("Invalid diagnostic queue");
      for (size_t i = 0; i < c.tail; ++i) {
        const auto ref = queue[i];
        if (ref.fit >= plan.fits || ref.node >= plan.layout[ref.fit].maximum_nodes)
          throw std::runtime_error("Invalid diagnostic node");
        const auto &node = reinterpret_cast<const gjxl_histogram_resident::Node *>(
            completed.data() + plan.layout[ref.fit].nodes)[ref.node];
        nodes << "{\"fit\":" << ref.fit << ",\"node\":" << ref.node
              << ",\"begin\":" << node.begin << ",\"end\":" << node.end
              << ",\"depth\":" << node.depth << ",\"property\":" << node.property
              << ",\"left\":" << node.left << ",\"right\":" << node.right << "}\n";
      }
    }
  }
private:
  MTL::Device *device_;
  std::span<const histogram_gpu_internal::ResidentHistogramRequest> requests_;
  size_t cohort_ = 0;
  bool active_ = false;
  std::filesystem::path directory_;
  std::set<size_t> selected_;
  std::ofstream log_, dispatches_, workload_;
};
} // namespace gjxl::frontier_diagnostic
