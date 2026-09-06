// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>

#include "core/overwrite_array.h"

int main() {
  gjxl::OverwriteArray<int32_t> empty;
  gjxl::OverwriteArray<int32_t> empty_copy(empty);
  if (empty.size() != 0 || empty.data() != nullptr ||
      empty.begin() != empty.end() || empty_copy.size() != 0) return EXIT_FAILURE;
  for (size_t count : {1u, 2u, 17u, 65536u, 196608u}) {
    gjxl::OverwriteArray<int32_t> values;
    values.ResetForOverwrite(count);
    for (size_t i = 0; i < count; ++i) values.data()[i] = static_cast<int32_t>(i) - 17;
    gjxl::OverwriteArray<int32_t> copy(values), assigned;
    assigned = values;
    if (copy.data() == values.data() || assigned.data() == values.data() ||
        !std::equal(values.begin(), values.end(), copy.begin(), copy.end()) ||
        !std::equal(values.begin(), values.end(), assigned.begin(), assigned.end())) return EXIT_FAILURE;
    values.data()[0] = 12345;
    if (copy.data()[0] != -17 || assigned.data()[0] != -17) return EXIT_FAILURE;
    const int32_t* before = values.data();
    auto* alias = &values;
    values = *alias;
    values = std::move(*alias);
    if (values.data() != before || values.size() != count || values.data()[0] != 12345) return EXIT_FAILURE;
    gjxl::OverwriteArray<int32_t> moved(std::move(values));
    if (values.size() != 0 || values.data() != nullptr || moved.data() != before) return EXIT_FAILURE;
    values = std::move(moved);
    if (moved.size() != 0 || moved.data() != nullptr || values.data() != before) return EXIT_FAILURE;
    try {
      values.ResetForOverwrite(std::numeric_limits<size_t>::max());
      return EXIT_FAILURE;
    } catch (const std::length_error&) {}
    if (values.data() != before || values.size() != count || values.data()[0] != 12345) return EXIT_FAILURE;
    values.assign(count, -31);
    if (!std::all_of(values.begin(), values.end(), [](int32_t value) { return value == -31; })) return EXIT_FAILURE;
    values = empty;
    if (values.size() != 0 || values.data() != nullptr) return EXIT_FAILURE;
    copy.ResetForOverwrite(0);
    if (copy.size() != 0 || copy.data() != nullptr) return EXIT_FAILURE;
  }
  std::cout << "Overwrite-array ownership and failure tests passed.\n" << std::flush;
  return EXIT_SUCCESS;
}
