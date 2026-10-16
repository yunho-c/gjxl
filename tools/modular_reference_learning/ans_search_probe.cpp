// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Deterministic model/bitstream and allocation-count comparison across builds.
#include "codestream/ans_internal.h"
#include "codestream/entropy_storage_plan.h"
#include "codestream/profile_internal.h"
#include "core/thread_budget.h"
#include "modular_allocation_test_util.h"
#include <fstream>
#include <iomanip>

using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;

void Ok(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
int main(int argc, char** argv) try {
  if (argc != 2) throw std::runtime_error("ans_search_probe OUT_DIRECTORY");
  const std::filesystem::path out = argv[1];
  if (std::filesystem::exists(out)) throw std::runtime_error("Output already exists");
  std::filesystem::create_directories(out);
  std::ofstream report(out / "results.jsonl");
  report.exceptions(std::ios::failbit | std::ios::badbit);
  for (size_t fixture = 0; fixture < 7; ++fixture) {
    std::array<std::vector<EntropyToken>, 3> sections;
    for (size_t i = 0; i < (fixture == 0 ? 0 : 8193); ++i) {
      const uint32_t value = fixture == 1 ? 0 : fixture == 2 ? uint32_t(i % 2)
        : fixture == 3 ? uint32_t((i * 19 + i / 31) % 256)
        : fixture == 4 ? uint32_t((i * 97 + i / 11) % 65536)
        : fixture == 5 ? uint32_t(i * 987654321u) : uint32_t((i / 127) % 9);
      sections[i % 2].push_back({uint32_t(i % 12), value});
    }
    std::array<EntropyTokenStreamView, 3> views;
    for (size_t s = 0; s < views.size(); ++s)
      views[s] = EntropyTokenStreamView::Interleaved(sections[s]);
    EntropyCode prefix;
    Ok(OptimizeEntropyCode(views, {.context_count = 12}, &prefix));
    for (size_t mode = 0; mode < 4; ++mode) {
      for (auto selection : {AnsSelectionMode::kExact, AnsSelectionMode::kEstimated}) {
        if ((mode == 1 || mode == 2) && selection == AnsSelectionMode::kEstimated) continue;
        const std::array policies{EntropyStoragePolicy::kAnsFromPrefix,
          EntropyStoragePolicy::kBalancedAns, EntropyStoragePolicy::kHighDensityAns,
          EntropyStoragePolicy::kRateOptimizedAns};
        const std::array modes{DirectAnsEntropyMode::kBalanced,
          DirectAnsEntropyMode::kHighDensity, DirectAnsEntropyMode::kRateOptimized};
        EntropyOptimizationStoragePlan plan;
        Ok(ComputeEntropyOptimizationStoragePlan({.policy = policies[mode],
          .tokens = sections[0].size() + sections[1].size(), .contexts = 12,
          .sections = views.size(), .return_cost = true}, &plan));
        ResourceBudget budget;
        ResourceReservation reservation;
        Ok(budget.Reserve(plan.working.peak_bytes, &reservation));
        EntropyCode model;
        EntropyCodeCost cost;
        EntropyWorkProfile profile;
        size_t allocations;
        {
          ResourceContextScope context({&reservation, ResourceClass::kSerializer});
          thread_budget_internal::EncodeScope serial(1);
          allocations = gjxl_test::CountManagedAllocations([&] {
            return mode == 0
              ? OptimizeAnsEntropyCode(views, prefix, &model, &cost, &profile, selection)
              : OptimizeDirectAnsEntropyCode(views, {.context_count = 12}, modes[mode - 1],
                                            &model, &cost, &profile, selection);
          });
        }
        // Search scratch has died. Exercise all retained histograms and tables.
        BitWriter writer;
        Ok(WriteEntropyCode(model, &writer));
        for (auto view : views) Ok(WriteTokenStream(view, model, &writer));
        const std::string name = std::to_string(fixture) + "-" + std::to_string(mode) + "-" +
          std::to_string(selection == AnsSelectionMode::kEstimated);
        std::ofstream bytes(out / (name + ".bin"), std::ios::binary);
        bytes.exceptions(std::ios::failbit | std::ios::badbit);
        const auto data = writer.padded_bytes();
        bytes.write(reinterpret_cast<const char*>(data.data()), data.size());
        report << "{\"case\":" << std::quoted(name) << ",\"allocations\":" << allocations
               << ",\"bits\":" << writer.bits_written()
               << ",\"configs\":" << profile.ans_uint_config_candidate_count
               << ",\"histograms\":" << profile.ans_histogram_candidate_count
               << ",\"peak_backing\":" << budget.snapshot().peak_backing_bytes
               << ",\"plan\":" << plan.working.peak_bytes << "}\n";
        model = {};
        cost = {};
        reservation.Reset();
        if (budget.snapshot().committed_bytes()) throw std::runtime_error("Leaked model storage");
      }
    }
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
