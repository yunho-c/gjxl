// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <span>

#include "gpu_profile_json.h"
#include "workflow_profile.h"

namespace gjxl::benchmark {

[[nodiscard]] inline std::string_view WorkflowEntropyBehaviorName(
    VarDctEntropyBehavior behavior) {
  switch (behavior) {
    case VarDctEntropyBehavior::kBalanced: return "balanced";
    case VarDctEntropyBehavior::kHighDensity: return "high-density";
    case VarDctEntropyBehavior::kMaximumCompression: return "maximum";
    case VarDctEntropyBehavior::kRateOptimized: return "rate-optimized";
  }
  return "invalid";
}

inline void WriteWorkflowPhaseNanoseconds(
    std::ostream& output, const codestream_internal::VarDctEncodingProfile& profile) {
  const auto phases = WorkflowProfileValues(profile);
  output << '{';
  for (size_t i = 0; i < phases.size(); ++i) {
    if (i != 0) output << ", ";
    output << '"' << kWorkflowProfileNames[i] << "\": " << phases[i];
  }
  output << '}';
}

inline void WriteWorkflowEntropyCoding(
    std::ostream& output, const codestream_internal::VarDctCodestreamProfile& cs) {
  output << "{\"dc\": \"" << (cs.dc_entropy_is_ans ? "ans" : "prefix")
         << "\", \"ac\": \"" << (cs.ac_entropy_is_ans ? "ans" : "prefix")
         << "\", \"coefficient_order\": \""
         << (cs.selected_coefficient_order_mask == 0 ? "none"
               : cs.coefficient_order_entropy_is_ans ? "ans" : "prefix")
         << "\"}";
}

// Uses the same run options as the sibling GPU export, but a separate schema
// and file: these are host elapsed/work durations, not device timestamps.
inline void WriteCudaWorkflowSamples(
    const std::filesystem::path& destination,
    const GpuProfileJsonOptions& options,
    std::string_view device,
    std::span<const RawWorkflowWorkload> workloads) {
  WriteProfileJsonFile(destination, [&](std::ostream& output) {
    output << "{\n  \"schema_version\": 1,\n"
           << "  \"scope\": \"cuda-public-workflow-host-profile\",\n"
           << "  \"timing_semantics\": \"instrumented-workflow-elapsed\",\n"
           << "  \"substage_work_timing\": \"aggregate-worker-time\",\n"
           << "  \"serializer_schedule\": \"detailed-profile-original-readiness\",\n"
           << "  \"gpu_profile_mode\": \""
           << GpuProfilingModeName(options.gpu_profiling_mode) << "\",\n"
           << "  \"device\": \"" << JsonEscape(device) << "\",\n"
           << "  \"gpu_aq\": \"" << JsonEscape(options.gpu_aq) << "\",\n"
           << "  \"distance\": " << std::setprecision(9) << options.butteraugli_target << ",\n"
           << "  \"effort\": " << options.effort << ",\n"
           << "  \"cpu_threads\": " << options.cpu_thread_count << ",\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"sample_count\": " << options.samples << ",\n"
           << "  \"collect_final_score\": "
           << (options.collect_final_butteraugli_score ? "true" : "false") << ",\n"
           << "  \"dc_quantization\": \"" << JsonEscape(options.dc_quantization) << "\",\n"
           << "  \"dc_prediction\": \"" << JsonEscape(options.dc_prediction) << "\",\n"
           << "  \"adaptive_dc_smoothing\": "
           << (options.adaptive_dc_smoothing ? (*options.adaptive_dc_smoothing ? "true" : "false") : "null")
           << ",\n  \"workloads\": [\n";
    for (size_t wi = 0; wi < workloads.size(); ++wi) {
      const auto& workload = workloads[wi];
      if (wi != 0) output << ",\n";
      output << "    {\"name\": \"" << JsonEscape(workload.workload)
             << "\", \"source_width\": " << workload.source_extent.width
             << ", \"source_height\": " << workload.source_extent.height
             << ", \"samples\": [\n";
      for (size_t si = 0; si < workload.samples.size(); ++si) {
        const auto& sample = workload.samples[si];
        const auto& profile = sample.profile;
        const auto& cs = profile.codestream;
        if (si != 0) output << ",\n";
        output << "      {\"sample_index\": " << sample.sample_index
               << ", \"backend\": \"" << JsonEscape(sample.backend)
               << "\", \"order\": \"" << JsonEscape(sample.order)
               << "\", \"encoded_bytes\": " << sample.encoded_bytes
               << ", \"peak_cpu_participants\": " << profile.peak_cpu_participants
               << ", \"ac_coefficient_bytes\": " << profile.ac_coefficient_bytes
               << ", \"ac_storage_bytes\": " << profile.ac_storage_bytes
               << ", \"phase_nanoseconds\": ";
        WriteWorkflowPhaseNanoseconds(output, profile);
        // Codestream total is internal to the enclosing workflow phase and must
        // not be summed with it. Keep it separately for accounting/overhead checks.
        output << ", \"codestream_total_nanoseconds\": " << cs.total_nanoseconds
               << ", \"serializer_counters\": {";
        bool first = true;
        const auto counter = [&](std::string_view name, auto value) {
          if (!first) output << ", ";
          first = false;
          output << '"' << name << "\": " << value;
        };
        counter("dc_sample_count", cs.dc_sample_count);
        counter("dc_leaf_count", cs.dc_leaf_count);
        counter("dc_context_count", cs.dc_context_count);
        counter("coefficient_tokenization_pass_count", cs.coefficient_tokenization_pass_count);
        counter("coefficient_token_count", cs.coefficient_token_count);
        counter("coefficient_context_materialization_count", cs.coefficient_context_materialization_count);
        counter("coefficient_materialized_token_count", cs.coefficient_materialized_token_count);
        counter("ans_uint_config_candidate_count", cs.entropy_work.ans_uint_config_candidate_count);
        counter("ans_histogram_candidate_count", cs.entropy_work.ans_histogram_candidate_count);
        counter("ans_alphabet_width_candidate_count", cs.entropy_work.ans_alphabet_width_candidate_count);
        counter("entropy_model_bits", cs.entropy_model_bits);
        counter("entropy_token_bits", cs.entropy_token_bits);
        counter("dc_entropy_clusters", cs.dc_entropy_clusters);
        counter("ac_entropy_clusters", cs.ac_entropy_clusters);
        counter("natural_candidate_bytes", cs.natural_candidate_bytes);
        counter("custom_order_candidate_bytes", cs.custom_order_candidate_bytes);
        counter("balanced_candidate_bytes", cs.balanced_candidate_bytes);
        counter("rate_candidate_bytes", cs.rate_candidate_bytes);
        counter("selected_coefficient_order_mask", cs.selected_coefficient_order_mask);
        counter("block_context_candidate_count", cs.block_context_candidate_count);
        counter("compact_block_context_candidate_bytes", cs.compact_block_context_candidate_bytes);
        counter("selected_block_context_candidate_index", cs.selected_block_context_candidate_index);
        counter("selected_block_context_count", cs.selected_block_context_count);
        counter("selected_block_context_qf_threshold_count", cs.selected_block_context_qf_threshold_count);
        output << "}, \"entropy_behavior\": \""
               << WorkflowEntropyBehaviorName(cs.entropy_behavior)
               << "\", \"coefficient_order_behavior\": \""
               << (cs.coefficient_order_behavior == VarDctCoefficientOrderBehavior::kFull
                     ? "full" : "effort7-dct8-sampled")
               << "\", \"selected_balanced_fallback\": "
               << (cs.selected_balanced_fallback ? "true" : "false")
               << ", \"entropy_coding\": ";
        WriteWorkflowEntropyCoding(output, cs);
        output << '}';
      }
      output << "\n    ]}";
    }
    output << "\n  ]\n}\n";
  });
}

// Schema-17 run metadata. Keep this adapter separate from CUDA's schema-1
// layout so existing Metal analysis scripts retain their field contract.
struct MetalWorkflowProfileJsonOptions {
  std::string_view scope;
  std::string_view validation;
  std::string_view implementation;
  std::string_view ac_residual_inverse;
  std::string_view gpu_aq;
  bool collect_final_butteraugli_score = false;
  std::string_view density;
  std::string_view compression;
  float butteraugli_target = 1.2f;
  int32_t effort = 7;
  size_t cpu_thread_count = 0;
  size_t warmups = 0;
  size_t samples = 0;
  std::optional<bool> adaptive_epf_sharpness;
};

inline void WriteMetalWorkflowSamples(
    const std::filesystem::path& destination,
    const MetalWorkflowProfileJsonOptions& options,
    std::span<const RawWorkflowWorkload> workloads) {
  WriteProfileJsonFile(destination, [&](std::ostream& output) {
    output << "{\n"
           << "  \"schema_version\": 17,\n"
           << "  \"substage_work_timing\": \"aggregate-worker-time\",\n"
           << "  \"scope\": \"" << JsonEscape(options.scope)
           << "\",\n"
           << "  \"validation\": \""
           << JsonEscape(options.validation) << "\",\n"
           << "  \"implementation\": \""
           << JsonEscape(options.implementation) << "\",\n"
           << "  \"ac_residual_inverse\": \""
           << JsonEscape(options.ac_residual_inverse) << "\",\n"
           << "  \"gpu_aq\": \"" << JsonEscape(options.gpu_aq)
           << "\",\n"
           << "  \"collect_final_score\": "
           << (options.collect_final_butteraugli_score ? "true" : "false")
           << ",\n"
           << "  \"density\": \""
           << JsonEscape(options.density)
           << "\",\n"
           << "  \"compression\": \""
           << JsonEscape(options.compression)
           << "\",\n"
           << "  \"distance\": " << std::setprecision(9)
           << options.butteraugli_target << ",\n"
           << "  \"effort\": " << options.effort << ",\n"
           << "  \"cpu_threads\": " << options.cpu_thread_count << ",\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"sample_count\": " << options.samples << ",\n";
    if (options.adaptive_epf_sharpness.has_value()) {
      output << "  \"adaptive_epf_sharpness\": "
             << (*options.adaptive_epf_sharpness ? "true" : "false") << ",\n";
    }
    output << "  \"workloads\": [\n";
    for (size_t workload_index = 0; workload_index < workloads.size();
         ++workload_index) {
      const RawWorkflowWorkload& workload = workloads[workload_index];
      output << "    {\n"
             << "      \"name\": \"" << JsonEscape(workload.workload)
             << "\",\n"
             << "      \"source_width\": " << workload.source_extent.width
             << ",\n"
             << "      \"source_height\": " << workload.source_extent.height
             << ",\n"
             << "      \"codestream_comparison\": \""
             << JsonEscape(workload.codestream_comparison) << "\",\n"
             << "      \"samples\": [\n";
      for (size_t sample_index = 0; sample_index < workload.samples.size();
           ++sample_index) {
        const RawWorkflowSample& sample = workload.samples[sample_index];
        const auto& profile = sample.profile;
        const auto& cs = profile.codestream;
        output << "        {\"sample_index\": " << sample.sample_index
               << ", \"backend\": \"" << JsonEscape(sample.backend)
               << "\", \"peak_cpu_participants\": "
               << profile.peak_cpu_participants
               << ", \"entropy_behavior\": \""
               << WorkflowEntropyBehaviorName(cs.entropy_behavior)
               << "\", \"entropy_search\": {\"uint_configs\": "
               << cs.entropy_work.ans_uint_config_candidate_count
               << ", \"histograms\": "
               << cs.entropy_work.ans_histogram_candidate_count
               << ", \"alphabet_widths\": "
               << cs.entropy_work.ans_alphabet_width_candidate_count << "}"
               << ", \"encoded_bytes\": " << sample.encoded_bytes
               << ", \"entropy_bits\": {\"model\": "
               << cs.entropy_model_bits << ", \"tokens\": "
               << cs.entropy_token_bits << "}"
               << ", \"entropy_clusters\": {\"dc\": "
               << cs.dc_entropy_clusters << ", \"ac\": "
               << cs.ac_entropy_clusters << "}"
               << ", \"entropy_coding\": ";
        WriteWorkflowEntropyCoding(output, cs);
        output
               << ", \"coefficient_order\": {\"natural_bytes\": "
               << cs.natural_candidate_bytes << ", \"custom_bytes\": "
               << cs.custom_order_candidate_bytes
               << ", \"selected_mask\": "
               << cs.selected_coefficient_order_mask << "}"
               << ", \"block_context\": {\"candidate_count\": "
               << cs.block_context_candidate_count
               << ", \"compact_bytes\": "
               << cs.compact_block_context_candidate_bytes
               << ", \"selected_index\": "
               << cs.selected_block_context_candidate_index
               << ", \"selected_contexts\": "
               << cs.selected_block_context_count
               << ", \"qf_thresholds\": "
               << cs.selected_block_context_qf_threshold_count << "}"
               << ", \"ac_tokenization\": {\"path\": \""
               << (cs.entropy_behavior ==
                     gjxl::VarDctEntropyBehavior::kMaximumCompression
                     ? "template" : "direct")
               << "\", \"pass_count\": "
               << cs.coefficient_tokenization_pass_count
               << ", \"tokens\": "
               << cs.coefficient_token_count
               << ", \"context_materialization_count\": "
               << cs.coefficient_context_materialization_count
               << ", \"materialized_tokens\": "
               << cs.coefficient_materialized_token_count << "}"
               << ", \"final_score\": ";
        if (sample.final_score.has_value()) {
          output << std::setprecision(17) << *sample.final_score;
        } else {
          output << "null";
        }
        output << ", \"phase_nanoseconds\": ";
        WriteWorkflowPhaseNanoseconds(output, profile);
        output << '}';
        if (sample_index + 1 != workload.samples.size()) {
          output << ',';
        }
        output << '\n';
      }
      output << "      ]\n"
             << "    }";
      if (workload_index + 1 != workloads.size()) {
        output << ',';
      }
      output << '\n';
    }
    output << "  ]\n}\n";
  });
}

// Reject aliases before encoding or writing either output. Existing hard links
// and symlinks are covered by equivalent(); canonicalization handles new paths.
inline bool ProfilePathsAlias(const std::filesystem::path& left,
                              const std::filesystem::path& right) {
  if (left.empty() || right.empty()) return false;
  std::error_code error;
  if (std::filesystem::equivalent(left, right, error) && !error) return true;
  const auto a = std::filesystem::weakly_canonical(std::filesystem::absolute(left));
  const auto b = std::filesystem::weakly_canonical(std::filesystem::absolute(right));
#if defined(_WIN32)
  return CompareStringOrdinal(a.c_str(), -1, b.c_str(), -1, TRUE) == CSTR_EQUAL;
#else
  return a == b;
#endif
}

}  // namespace gjxl::benchmark
