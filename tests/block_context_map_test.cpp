// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/block_context_map.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "codestream/simple_ac_context.h"

namespace {

bool CheckDefaultMap() {
  const gjxl::SimpleBlockContextMap map =
    gjxl::DefaultSimpleBlockContextMap();
  if (!gjxl::ValidateSimpleBlockContextMap(map).ok() ||
      map.num_contexts != 4 || map.ac_context_count() != 1980 ||
      map.context_map.size() !=
        gjxl::codestream_internal::kSimpleBlockContextMap.size()) {
    std::cerr << "Default block-context map is invalid\n";
    return false;
  }
  for (size_t strategy = 0; strategy < gjxl::kAcStrategyCount; ++strategy) {
    for (size_t channel = 0; channel < 3; ++channel) {
      uint32_t context = 99;
      const auto type = static_cast<gjxl::AcStrategyType>(strategy);
      if (!gjxl::SimpleBlockContext(
            map, type, channel, 1, &context).ok() ||
          context != gjxl::codestream_internal::SimpleBlockContext(
            type, channel)) {
        std::cerr << "Default block-context lookup changed\n";
        return false;
      }
    }
  }
  return true;
}

bool CheckQuantizationSplit() {
  const gjxl::SimpleBlockContextMap compact =
    gjxl::DefaultSimpleBlockContextMap();
  gjxl::SimpleBlockContextMap split;
  split.qf_thresholds = {29};
  split.num_contexts = 8;
  split.context_map.resize(2 * compact.context_map.size());
  for (size_t row_order = 0; row_order < compact.context_map.size();
       ++row_order) {
    split.context_map[2 * row_order] = compact.context_map[row_order];
    split.context_map[2 * row_order + 1] =
      static_cast<uint8_t>(compact.context_map[row_order] + 4);
  }
  uint32_t low = 99;
  uint32_t boundary = 99;
  uint32_t high = 99;
  if (!gjxl::ValidateSimpleBlockContextMap(split).ok() ||
      split.ac_context_count() != 3960 ||
      !gjxl::SimpleBlockContext(
        split, gjxl::AcStrategyType::kDct8, 1, 1, &low).ok() ||
      !gjxl::SimpleBlockContext(
        split, gjxl::AcStrategyType::kDct8, 1, 29, &boundary).ok() ||
      !gjxl::SimpleBlockContext(
        split, gjxl::AcStrategyType::kDct8, 1, 30, &high).ok() ||
      low != 0 || boundary != 0 || high != 4) {
    std::cerr << "Raw-quant block-context split is incorrect\n";
    return false;
  }
  return true;
}

bool CheckTwoChannelMap() {
  const gjxl::SimpleBlockContextMap map =
    gjxl::TwoChannelSimpleBlockContextMap();
  uint32_t y_context = 99;
  uint32_t x_context = 99;
  uint32_t b_context = 99;
  if (!gjxl::ValidateSimpleBlockContextMap(map).ok() ||
      map.num_contexts != 2 ||
      !gjxl::SimpleBlockContext(
        map, gjxl::AcStrategyType::kDct32x32, 1, 256, &y_context).ok() ||
      !gjxl::SimpleBlockContext(
        map, gjxl::AcStrategyType::kDct8, 0, 1, &x_context).ok() ||
      !gjxl::SimpleBlockContext(
        map, gjxl::AcStrategyType::kAfv3, 2, 127, &b_context).ok() ||
      y_context != 0 || x_context != 1 || b_context != 1) {
    std::cerr << "Two-channel block-context map is invalid\n";
    return false;
  }
  return true;
}

bool CheckInvalidMapsAreAtomic() {
  gjxl::SimpleBlockContextMap invalid =
    gjxl::DefaultSimpleBlockContextMap();
  invalid.qf_thresholds = {0};
  uint32_t context = 77;
  if (gjxl::ValidateSimpleBlockContextMap(invalid).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      gjxl::SimpleBlockContext(
        invalid, gjxl::AcStrategyType::kDct8, 0, 1, &context).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      context != 77) {
    std::cerr << "Invalid block-context map changed its output\n";
    return false;
  }

  invalid = gjxl::DefaultSimpleBlockContextMap();
  invalid.context_map[0] = invalid.num_contexts;
  if (gjxl::ValidateSimpleBlockContextMap(invalid).code() !=
      gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Out-of-range block context was accepted\n";
    return false;
  }
  invalid = gjxl::DefaultSimpleBlockContextMap();
  for (uint8_t& label : invalid.context_map) {
    if (label == 0) label = 2;
  }
  if (gjxl::ValidateSimpleBlockContextMap(invalid).code() !=
      gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Noncanonical block-context labels were accepted\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  return CheckDefaultMap() && CheckTwoChannelMap() &&
      CheckQuantizationSplit() &&
      CheckInvalidMapsAreAtomic()
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
}
