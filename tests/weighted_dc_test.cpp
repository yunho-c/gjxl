// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Native weighted predictor boundary tests against pinned libjxl's predictor.
#include "codestream/weighted_dc.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include <array>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
int main() {
  try {
    size_t cases = 0;
    for (auto extent : std::array<gjxl::Extent2D, 9>{{{1, 1},
                                                      {1, 256},
                                                      {256, 1},
                                                      {2, 2},
                                                      {3, 7},
                                                      {17, 19},
                                                      {255, 256},
                                                      {256, 255},
                                                      {256, 256}}}) {
      for (unsigned pattern = 0; pattern < 5; ++pattern) {
        const size_t stride = extent.width + 5;
        std::array<std::vector<int32_t>, 3> storage;
        gjxl::ConstImage3I32View input;
        uint32_t random = 0x31415926;
        for (size_t c = 0; c < 3; ++c) {
          storage[c].assign(stride * extent.height, 1234567);
          for (size_t y = 0; y < extent.height; ++y)
            for (size_t x = 0; x < extent.width; ++x) {
              random = random * 1664525 + 1013904223;
              int32_t value = 0;
              if (pattern == 1)
                value = static_cast<int32_t>(c) * 19 - 11;
              if (pattern == 2)
                value =
                    static_cast<int32_t>((x * 5 + y * 7 + c * 13) % 1000) - 500;
              if (pattern == 3)
                value = static_cast<int32_t>(random >> 16) - 32768;
              if (pattern == 4)
                value = ((x + y + c) & 1) ? 32767 : -32768;
              storage[c][y * stride + x] = value;
            }
          input.plane[c] = {storage[c].data(), extent, stride};
        }
        gjxl::codestream_internal::Storage<gjxl::EntropyToken> actual;
        Check(gjxl::codestream_internal::TokenizeWeightedDcGroup(input, &actual)
                  .ok(),
              "native tokenization failed");
        size_t index = 0;
        for (size_t c : {size_t{1}, size_t{0}, size_t{2}}) {
          jxl::weighted::Header header;
          jxl::weighted::PredictorMode(
              0, &header); // Pin the serializer's default preset explicitly.
          jxl::weighted::State state(header, extent.width, extent.height);
          jxl::Properties properties(1);
          const auto plane = input.plane[c];
          for (size_t y = 0; y < extent.height; ++y)
            for (size_t x = 0; x < extent.width; ++x) {
              const int64_t w = x   ? plane.Row(y)[x - 1]
                                : y ? plane.Row(y - 1)[x]
                                    : 0;
              const int64_t n = y ? plane.Row(y - 1)[x] : w;
              const int64_t nw = x && y ? plane.Row(y - 1)[x - 1] : w;
              const int64_t ne =
                  y && x + 1 < extent.width ? plane.Row(y - 1)[x + 1] : n;
              const int64_t nn = y > 1 ? plane.Row(y - 2)[x] : n;
              const auto prediction = state.Predict<true>(
                  x, y, extent.width, n, w, ne, nw, nn, &properties, 0);
              const int64_t residual = int64_t{plane.Row(y)[x]} - prediction;
              if (actual.at(index).value !=
                  gjxl::PackSigned(static_cast<int32_t>(residual))) {
                std::cerr << extent.width << "x" << extent.height
                          << " pattern=" << pattern << " c=" << c << " x=" << x
                          << " y=" << y << " ref=" << prediction
                          << " packed=" << actual.at(index).value
                          << " p1=" << header.p1C << " w0=" << header.w[0]
                          << "\n";
                throw std::runtime_error("residual differs from libjxl");
              }
              Check(actual.at(index).context >= 11 &&
                        actual.at(index).context < 45,
                    "DC context overlaps metadata");
              state.UpdateErrors(plane.Row(y)[x], x, y, extent.width);
              ++index;
            }
        }
        Check(index == actual.size(), "token count changed");
        auto repeat = [&] {
          gjxl::codestream_internal::Storage<gjxl::EntropyToken> result;
          Check(
              gjxl::codestream_internal::TokenizeWeightedDcGroup(input, &result)
                  .ok(),
              "repeat failed");
          return result;
        };
        auto concurrent = std::async(std::launch::async, repeat);
        Check(repeat() == actual && concurrent.get() == actual,
              "state leaks between calls");
        ++cases;
      }
    }
    gjxl::codestream_internal::Storage<gjxl::EntropyToken> sentinel = {{12,
                                                                        345}},
                                                           output = sentinel;
    Check(
        !gjxl::codestream_internal::TokenizeWeightedDcGroup({}, &output).ok() &&
            output == sentinel,
        "invalid geometry changed output");
    Check(!gjxl::codestream_internal::TokenizeWeightedDcGroup({}, nullptr).ok(),
          "null output accepted");
    std::array<int32_t, 2> extreme = {std::numeric_limits<int32_t>::min(),
                                      std::numeric_limits<int32_t>::max()};
    gjxl::ConstImage3I32View input;
    for (auto &plane : input.plane)
      plane = {extreme.data(), {2, 1}, 2};
    Check(!gjxl::codestream_internal::TokenizeWeightedDcGroup(input, &output)
                  .ok() &&
              output == sentinel,
          "unrepresentable state accepted or output changed");
    Check(!gjxl::codestream_internal::UseWeightedDcTree(output).ok() &&
              output == sentinel,
          "invalid tree changed output");
    Check(!gjxl::codestream_internal::UseWeightedDcTree({}).ok(),
          "null tree accepted");
    std::cout << cases
              << " strided boundary/oracle cases plus atomic rejection checks "
                 "passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
