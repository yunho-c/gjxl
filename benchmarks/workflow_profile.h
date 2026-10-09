// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "codestream/workflow_internal.h"

namespace gjxl::benchmark {

// Shared Metal/CUDA names. Timings ending in _work are aggregate worker time,
// not additive wall-clock phases; all other fields retain their nested scope.
inline constexpr std::array<std::string_view, 45> kWorkflowProfileNames = {
    "total",
    "input_preparation",
    "input_geometry_and_storage",
    "input_color_transform",
    "input_matrix_scale_stats",
    "input_resident_preparation",
    "input_quantization_preparation",
    "backend_selection",
    "quantization_pipeline",
    "codestream_encoding",
    "summary_assembly",
    "codestream_validation",
    "codestream_dc_tokenization",
    "codestream_ac_tokenization",
    "codestream_block_context_map_work",
    "codestream_coefficient_order_work",
    "codestream_coefficient_tokenization_work",
    "codestream_coefficient_context_materialization_work",
    "codestream_entropy_optimization",
    "codestream_entropy_prefix_histogram_build_work",
    "codestream_entropy_prefix_histogram_cost_work",
    "codestream_entropy_prefix_clustering_work",
    "codestream_entropy_prefix_code_build_work",
    "codestream_entropy_prefix_value_collection_work",
    "codestream_entropy_prefix_config_search_work",
    "codestream_entropy_prefix_exact_measurement_work",
    "codestream_entropy_ans_prefix_validation_work",
    "codestream_entropy_ans_value_collection_work",
    "codestream_entropy_ans_value_aggregation_work",
    "codestream_entropy_ans_prepared_value_validation_work",
    "codestream_entropy_ans_uint_config_work",
    "codestream_entropy_ans_histogram_build_work",
    "codestream_entropy_ans_model_build_work",
    "codestream_entropy_ans_token_cost_work",
    "codestream_entropy_selection_work",
    "codestream_section_writing",
    "codestream_section_model_and_header_work",
    "codestream_section_token_write_work",
    "codestream_section_candidate_measure_work",
    "codestream_assembly",
    "codestream_assembly_candidate_selection",
    "codestream_assembly_section_size",
    "codestream_assembly_frame_header",
    "codestream_assembly_toc_and_sections",
    "codestream_assembly_output_copy",
};

using WorkflowProfileNanoseconds =
    std::array<uint64_t, kWorkflowProfileNames.size()>;

[[nodiscard]] inline WorkflowProfileNanoseconds WorkflowProfileValues(
    const gjxl::codestream_internal::VarDctEncodingProfile& profile) {
  return {
      profile.total_nanoseconds,
      profile.input_preparation_nanoseconds,
      profile.input_geometry_and_storage_nanoseconds,
      profile.input_color_transform_nanoseconds,
      profile.input_matrix_scale_stats_nanoseconds,
      profile.input_resident_preparation_nanoseconds,
      profile.input_quantization_preparation_nanoseconds,
      profile.backend_selection_nanoseconds,
      profile.quantization_pipeline_nanoseconds,
      profile.codestream_encoding_nanoseconds,
      profile.summary_assembly_nanoseconds,
      profile.codestream.validation_nanoseconds,
      profile.codestream.dc_tokenization_nanoseconds,
      profile.codestream.ac_tokenization_nanoseconds,
      profile.codestream.block_context_map_work_nanoseconds,
      profile.codestream.coefficient_order_work_nanoseconds,
      profile.codestream.coefficient_tokenization_work_nanoseconds,
      profile.codestream.coefficient_context_materialization_work_nanoseconds,
      profile.codestream.entropy_optimization_nanoseconds,
      profile.codestream.entropy_work.prefix_histogram_build_nanoseconds,
      profile.codestream.entropy_work.prefix_histogram_cost_nanoseconds,
      profile.codestream.entropy_work.prefix_clustering_nanoseconds,
      profile.codestream.entropy_work.prefix_code_build_nanoseconds,
      profile.codestream.entropy_work.prefix_value_collection_nanoseconds,
      profile.codestream.entropy_work.prefix_config_search_nanoseconds,
      profile.codestream.entropy_work.prefix_exact_measurement_nanoseconds,
      profile.codestream.entropy_work.ans_prefix_validation_nanoseconds,
      profile.codestream.entropy_work.ans_value_collection_nanoseconds,
      profile.codestream.entropy_work.ans_value_aggregation_nanoseconds,
      profile.codestream.entropy_work
        .ans_prepared_value_validation_nanoseconds,
      profile.codestream.entropy_work.ans_uint_config_nanoseconds,
      profile.codestream.entropy_work.ans_histogram_build_nanoseconds,
      profile.codestream.entropy_work.ans_model_build_nanoseconds,
      profile.codestream.entropy_work.ans_token_cost_nanoseconds,
      profile.codestream.entropy_work.selection_nanoseconds,
      profile.codestream.section_writing_nanoseconds,
      profile.codestream.section_writing_work.model_and_header_nanoseconds,
      profile.codestream.section_writing_work.token_write_nanoseconds,
      profile.codestream.section_writing_work.candidate_measure_nanoseconds,
      profile.codestream.assembly_nanoseconds,
      profile.codestream.assembly.candidate_selection_nanoseconds,
      profile.codestream.assembly.section_size_nanoseconds,
      profile.codestream.assembly.frame_header_nanoseconds,
      profile.codestream.assembly.toc_and_sections_nanoseconds,
      profile.codestream.assembly.output_copy_nanoseconds,
  };
}

}  // namespace gjxl::benchmark
