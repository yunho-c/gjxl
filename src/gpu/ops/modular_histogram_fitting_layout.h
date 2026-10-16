// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
namespace gjxl_histogram_fitting {
using Word = uint;
#else
#include <cstdint>
namespace gjxl_histogram_fitting {
using Word = uint32_t;
#endif

enum : Word {
  kStaticConstant = 0,
  kStatic = 1,
  kNonstatic = 2,
  kNonstaticWithoutWeighted = 3,
  kCategories = 4,
  kRowIndexError = 1,
  kPopulationError = 2,
  kParentSummaryError = 4,
  kArithmeticError = 8,
};

// Byte offsets are relative to the prepared arena, including its descriptors.
struct FitDescriptor {
  Word rows, population, predictors, properties;
  Word static_properties, source_id, weights_offset, reserved;
  Word predictor_ids[14], property_ids[16], bins[16];
  Word residual_offsets[14], property_offsets[16];
};
// Offsets are relative to the dispatch arena; result_first is an element index.
struct NodeDescriptor {
  Word fit, node_id, rows, symbols;
  Word indices_offset, parent_counts_offset, parent_extra_offset, result_first;
  Word inherited_constants, parent_predictor, predictor_penalties_offset,
      reserved;
};
// Canonical nonconstant property work; reusable scratch slots are assigned by
// the backend's bounded waves rather than retained in this descriptor.
struct TaskDescriptor {
  Word node, property;
};
// low_rows counts row entries, not their weights. Predictor fields are wire
// IDs. Invalid candidates have valid=0, both costs=FLT_MAX, and all other
// fields zero.
struct Candidate {
  Word bin, low_rows, left_predictor, right_predictor;
  float left_cost, right_cost;
  Word valid, reserved;
};
struct PropertyResult {
  Word source_id, node_id, property_index, is_constant;
  Candidate winners[kCategories];
};
struct CandidateCosts {
  float low, high;
};
struct SideWinner {
  float cost, penalty;
  Word predictor;
};
} // namespace gjxl_histogram_fitting
