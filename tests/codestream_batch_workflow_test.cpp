// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "codestream/batch_workflow.h"
#include "codestream/workflow.h"
#include "core/image.h"

namespace {

constexpr gjxl::Extent2D kExtent{32, 24};
constexpr size_t kStride = kExtent.width + 3;

struct ImageStorage {
  explicit ImageStorage(size_t phase) {
    for (std::vector<float>& values : plane) {
      values.assign(kStride * kExtent.height, -123.0f);
    }
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        const float fx = static_cast<float>(x) /
          static_cast<float>(kExtent.width - 1);
        const float fy = static_cast<float>(y) /
          static_cast<float>(kExtent.height - 1);
        const float texture = static_cast<float>(
          (11 * x + 7 * y + 3 * phase) % 17) / 160.0f;
        plane[0][y * kStride + x] = 0.03f + 0.72f * fx + texture;
        plane[1][y * kStride + x] = 0.02f + 0.68f * fy + texture;
        plane[2][y * kStride + x] =
          0.04f + 0.31f * fx + 0.43f * fy + texture;
      }
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView View() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[1].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }

  std::array<std::vector<float>, 3> plane;
};

bool CheckBatchMatchesSequential() {
  constexpr size_t kImageCount = 5;
  std::array<ImageStorage, kImageCount> images = {
    ImageStorage(0), ImageStorage(1), ImageStorage(2), ImageStorage(3),
    ImageStorage(4)};
  std::vector<gjxl::VarDctBatchEncodingRequest> requests;
  requests.reserve(images.size());
  for (const ImageStorage& image : images) {
    requests.push_back({
      .linear_rgb = image.View(),
      .options = {
        .butteraugli_target = 1.0f,
        .backend = gjxl::VarDctBackendPreference::kCpu,
      },
    });
  }

  std::vector<std::vector<uint8_t>> expected_codestreams(images.size());
  std::vector<gjxl::VarDctEncodingSummary> expected_summaries(images.size());
  for (size_t index = 0; index < images.size(); ++index) {
    const gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
      requests[index].linear_rgb,
      requests[index].options,
      &expected_codestreams[index],
      &expected_summaries[index]);
    if (!status.ok()) {
      std::cerr << "Sequential reference " << index << " failed: "
                << status.message() << '\n';
      return false;
    }
  }

  std::unique_ptr<gjxl::VarDctBatchEncoder> encoder;
  gjxl::Status status = gjxl::VarDctBatchEncoder::Create(3, &encoder);
  if (!status.ok() || encoder == nullptr || encoder->max_in_flight() != 3) {
    std::cerr << "Unable to create three-worker batch encoder: "
              << status.message() << '\n';
    return false;
  }

  std::vector<gjxl::VarDctBatchEncodingResult> results;
  status = encoder->Encode(requests, &results);
  if (!status.ok() || results.size() != requests.size()) {
    std::cerr << "Batch scheduling failed: " << status.message() << '\n';
    return false;
  }
  for (size_t index = 0; index < results.size(); ++index) {
    if (!results[index].status.ok() ||
        results[index].codestream != expected_codestreams[index] ||
        results[index].summary != expected_summaries[index] ||
        results[index].timing.total_nanoseconds == 0) {
      std::cerr << "Batch result " << index
                << " did not match its sequential reference\n";
      return false;
    }
  }

  std::vector<gjxl::VarDctBatchEncodingResult> second_results;
  status = encoder->Encode(requests, &second_results);
  if (!status.ok() || second_results.size() != results.size()) {
    std::cerr << "Reused batch encoder failed\n";
    return false;
  }
  for (size_t index = 0; index < results.size(); ++index) {
    if (!second_results[index].status.ok() ||
        second_results[index].codestream != results[index].codestream ||
        second_results[index].summary != results[index].summary) {
      std::cerr << "Reused batch encoder changed result " << index << '\n';
      return false;
    }
  }
  return true;
}

bool CheckPerImageFailureAndEmptyBatch() {
  ImageStorage first(5);
  ImageStorage second(6);
  std::vector<gjxl::VarDctBatchEncodingRequest> requests = {
    {
      .linear_rgb = first.View(),
      .options = {
        .butteraugli_target = 1.0f,
        .backend = gjxl::VarDctBackendPreference::kCpu,
      },
    },
    {
      .linear_rgb = {},
      .options = {
        .butteraugli_target = 1.0f,
        .backend = gjxl::VarDctBackendPreference::kCpu,
      },
    },
    {
      .linear_rgb = second.View(),
      .options = {
        .butteraugli_target = 1.0f,
        .backend = gjxl::VarDctBackendPreference::kCpu,
      },
    },
  };

  std::unique_ptr<gjxl::VarDctBatchEncoder> encoder;
  gjxl::Status status = gjxl::VarDctBatchEncoder::Create(8, &encoder);
  if (!status.ok() || encoder == nullptr) {
    return false;
  }
  std::vector<gjxl::VarDctBatchEncodingResult> results;
  status = encoder->Encode(requests, &results);
  if (!status.ok() || results.size() != 3 ||
      !results[0].status.ok() ||
      results[1].status.code() != gjxl::StatusCode::kInvalidArgument ||
      !results[1].codestream.empty() ||
      !results[2].status.ok()) {
    std::cerr << "Per-image batch failure was not isolated\n";
    return false;
  }

  results.push_back({});
  status = encoder->Encode({}, &results);
  if (!status.ok() || !results.empty()) {
    std::cerr << "Empty batch did not commit an empty result\n";
    return false;
  }
  return true;
}

bool CheckInvalidDriverArguments() {
  std::unique_ptr<gjxl::VarDctBatchEncoder> encoder;
  gjxl::Status status = gjxl::VarDctBatchEncoder::Create(0, &encoder);
  if (status.code() != gjxl::StatusCode::kInvalidArgument || encoder) {
    std::cerr << "Zero-worker batch encoder was accepted\n";
    return false;
  }
  status = gjxl::VarDctBatchEncoder::Create(1, nullptr);
  if (status.code() != gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Null batch encoder output was accepted\n";
    return false;
  }
  status = gjxl::VarDctBatchEncoder::Create(1, &encoder);
  if (!status.ok() || encoder == nullptr ||
      encoder->Encode({}, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Null batch result output was accepted\n";
    return false;
  }
  return true;
}

bool CheckSharedMetalBackendIfAvailable() {
  ImageStorage image(7);
  const gjxl::VarDctEncodingOptions options = {
    .butteraugli_target = 1.2f,
    .backend = gjxl::VarDctBackendPreference::kMetal,
    .gpu_aq_mode =
      gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput,
  };
  std::vector<uint8_t> expected_codestream;
  gjxl::VarDctEncodingSummary expected_summary;
  gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), options, &expected_codestream, &expected_summary);
  if (status.code() == gjxl::StatusCode::kUnavailable) {
    std::cout << "Metal batch check skipped: " << status.message() << '\n';
    return true;
  }
  if (!status.ok()) {
    std::cerr << "Metal batch reference failed: "
              << status.message() << '\n';
    return false;
  }

  constexpr size_t kImageCount = 4;
  std::vector<gjxl::VarDctBatchEncodingRequest> requests(
    kImageCount, {.linear_rgb = image.View(), .options = options});
  std::unique_ptr<gjxl::VarDctBatchEncoder> encoder;
  status = gjxl::VarDctBatchEncoder::Create(kImageCount, &encoder);
  if (!status.ok() || encoder == nullptr) {
    return false;
  }
  std::vector<gjxl::VarDctBatchEncodingResult> results;
  status = encoder->Encode(requests, &results);
  if (!status.ok() || results.size() != kImageCount) {
    std::cerr << "Concurrent Metal batch scheduling failed\n";
    return false;
  }
  for (size_t index = 0; index < results.size(); ++index) {
    if (!results[index].status.ok() ||
        results[index].codestream != expected_codestream ||
        results[index].summary != expected_summary ||
        results[index].summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal) {
      std::cerr << "Concurrent Metal result " << index
                << " changed the single-image output\n";
      return false;
    }
  }
  return true;
}

bool CheckBoundedCudaBackendsIfAvailable() {
#if !GJXL_TEST_HAS_CUDA
  return true;
#else
  ImageStorage image(8);
  const gjxl::VarDctEncodingOptions options = {
    .butteraugli_target = 1.2f,
    .backend = gjxl::VarDctBackendPreference::kCuda,
    .gpu_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
  };
  std::vector<uint8_t> expected_codestream;
  gjxl::VarDctEncodingSummary expected_summary;
  gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), options, &expected_codestream, &expected_summary);
  if (status.code() == gjxl::StatusCode::kUnavailable) {
    std::cout << "CUDA batch check skipped: " << status.message() << '\n';
    return true;
  }
  if (!status.ok()) {
    std::cerr << "CUDA batch reference failed: "
              << status.message() << '\n';
    return false;
  }

  constexpr size_t kImageCount = 8;
  constexpr size_t kRepetitions = 4;
  std::vector<gjxl::VarDctBatchEncodingRequest> requests(
    kImageCount, {.linear_rgb = image.View(), .options = options});
  std::unique_ptr<gjxl::VarDctBatchEncoder> encoder;
  status = gjxl::VarDctBatchEncoder::Create(kImageCount, &encoder);
  if (!status.ok() || encoder == nullptr) {
    return false;
  }
  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    std::vector<gjxl::VarDctBatchEncodingResult> results;
    status = encoder->Encode(requests, &results);
    if (!status.ok() || results.size() != kImageCount) {
      std::cerr << "Concurrent CUDA batch scheduling failed\n";
      return false;
    }
    for (size_t index = 0; index < results.size(); ++index) {
      if (!results[index].status.ok() ||
          results[index].codestream != expected_codestream ||
          results[index].summary != expected_summary ||
          results[index].summary.execution_backend !=
            gjxl::VarDctExecutionBackend::kCuda) {
        std::cerr << "Concurrent CUDA result " << index
                  << " changed the single-image output\n";
        return false;
      }
    }
  }
  return true;
#endif
}

}  // namespace

int main() {
  if (!CheckBatchMatchesSequential() ||
      !CheckPerImageFailureAndEmptyBatch() ||
      !CheckInvalidDriverArguments() ||
      !CheckSharedMetalBackendIfAvailable() ||
      !CheckBoundedCudaBackendsIfAvailable()) {
    return EXIT_FAILURE;
  }
  std::cout << "Codestream batch workflow checks passed\n";
  return EXIT_SUCCESS;
}
