// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "butteraugli_fixtures.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gjxl::butteraugli_test {
namespace {

[[nodiscard]] uint32_t Xorshift32(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

[[nodiscard]] float RandomUnit(uint32_t *state) {
  return static_cast<float>(Xorshift32(state) >> 8) * (1.0f / 16777215.0f);
}

[[nodiscard]] float SrgbToLinear(uint8_t value) {
  const float encoded = static_cast<float>(value) * (1.0f / 255.0f);
  if (encoded <= 0.04045f) {
    return encoded * (1.0f / 12.92f);
  }
  return std::pow((encoded + 0.055f) * (1.0f / 1.055f), 2.4f);
}

void FillPattern(const FixtureSpec &spec, ImageStorage *reference,
                 ImageStorage *distorted) {
  const size_t width = spec.extent.width;
  const size_t height = spec.extent.height;
  const size_t stride = reference->stride();
  uint32_t random_state = spec.random_seed;
  if (random_state == 0) {
    random_state = 1;
  }

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      for (size_t channel = 0; channel < 3; ++channel) {
        const float fx = static_cast<float>(x);
        const float fy = static_cast<float>(y);
        float value = 0.0f;
        float error = 0.0f;
        switch (spec.kind) {
        case FixtureKind::kFlat:
          value = 0.18f + 0.04f * static_cast<float>(channel);
          if (x >= 10 && x < 22 && y >= 7 && y < 17) {
            error = 0.012f * static_cast<float>(channel + 1);
          }
          break;
        case FixtureKind::kTexture:
        case FixtureKind::kIdentityTexture:
          value = 0.25f +
                  0.11f * std::sin(0.41f * static_cast<float>(
                                               (channel + 1) * x + 2 * y)) +
                  0.07f * std::cos(0.27f * (3.0f * fx - fy));
          if (spec.kind == FixtureKind::kTexture) {
            error =
                0.018f *
                std::sin(0.73f * static_cast<float>(5 * x + 3 * y + channel));
          }
          break;
        case FixtureKind::kContrast:
          value = ((x / 4 + y / 4) & 1u) == 0 ? 0.02f : 0.92f;
          error = (x % 4 == 0 || y % 4 == 0) ? (value < 0.5f ? 0.035f : -0.035f)
                                             : 0.0f;
          break;
        case FixtureKind::kChromatic:
          value =
              channel == 0
                  ? 0.05f +
                        0.75f * fx /
                            static_cast<float>(std::max<size_t>(1, width - 1))
              : channel == 1
                  ? 0.08f +
                        0.65f * fy /
                            static_cast<float>(std::max<size_t>(1, height - 1))
                  : 0.72f - 0.55f * (fx + fy) /
                                static_cast<float>(
                                    std::max<size_t>(1, width + height - 2));
          error = channel == 0   ? 0.02f * std::sin(0.31f * fy)
                  : channel == 2 ? -0.018f * std::cos(0.29f * fx)
                                 : 0.0f;
          break;
        case FixtureKind::kGradient:
          value = std::clamp(
              0.03f +
                  0.72f * fx /
                      static_cast<float>(std::max<size_t>(1, width - 1)) +
                  0.17f * fy /
                      static_cast<float>(std::max<size_t>(1, height - 1)) +
                  0.025f * static_cast<float>(channel),
              0.0f, 1.0f);
          error = 0.011f * static_cast<float>(channel + 1) *
                  std::sin(0.37f * fx + 0.19f * fy);
          break;
        case FixtureKind::kIntermediate:
          value = std::clamp(
              0.08f + 0.021f * fx + 0.013f * fy +
                  0.09f * std::sin(0.43f *
                                   static_cast<float>((channel + 1) * x + y)) +
                  0.04f * std::cos(0.31f * (fx - 2.0f * fy)),
              -0.1f, 1.2f);
          error = 0.024f * std::sin(0.67f * static_cast<float>(3 * x + 5 * y +
                                                               channel)) +
                  ((x + 2 * y + channel) % 7 == 0 ? 0.013f : 0.0f);
          break;
        case FixtureKind::kImpulse:
          value = 0.08f + 0.02f * static_cast<float>(channel);
          if (channel == spec.impulse_channel && x == width / 2 &&
              y == height / 2) {
            error = 0.55f;
          }
          break;
        case FixtureKind::kRandomLow:
        case FixtureKind::kRandomNormal:
        case FixtureKind::kRandomWide: {
          float low = 0.0f;
          float high = 0.08f;
          if (spec.kind == FixtureKind::kRandomNormal) {
            high = 1.0f;
          } else if (spec.kind == FixtureKind::kRandomWide) {
            low = -0.25f;
            high = 1.5f;
          }
          value = low + (high - low) * RandomUnit(&random_state);
          error = (RandomUnit(&random_state) - 0.5f) *
                  (spec.kind == FixtureKind::kRandomLow ? 0.008f : 0.06f);
          break;
        }
        }
        float distorted_value = value + error;
        if (spec.kind == FixtureKind::kRandomLow) {
          distorted_value = std::clamp(distorted_value, 0.0f, 0.08f);
        } else if (spec.kind == FixtureKind::kRandomNormal) {
          distorted_value = std::clamp(distorted_value, 0.0f, 1.0f);
        } else if (spec.kind == FixtureKind::kRandomWide) {
          distorted_value = std::clamp(distorted_value, -0.25f, 1.5f);
        }
        reference->planes()[channel][y * stride + x] = value;
        distorted->planes()[channel][y * stride + x] = distorted_value;
      }
    }
  }
}

[[nodiscard]] std::string ReadPpmToken(std::istream *input) {
  std::string token;
  while (true) {
    *input >> std::ws;
    if (input->peek() != '#') {
      break;
    }
    input->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  *input >> token;
  if (!*input) {
    throw std::runtime_error("Malformed PPM header");
  }
  return token;
}

[[nodiscard]] ImageStorage LoadLinearPpm(const std::string &ppm_path) {
  std::ifstream input(ppm_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open flower PPM: " + ppm_path);
  }
  if (ReadPpmToken(&input) != "P6") {
    throw std::runtime_error("Flower fixture is not a binary RGB PPM");
  }
  const size_t width = std::stoul(ReadPpmToken(&input));
  const size_t height = std::stoul(ReadPpmToken(&input));
  const unsigned long maximum = std::stoul(ReadPpmToken(&input));
  if (width == 0 || height == 0 || maximum != 255) {
    throw std::runtime_error("Flower PPM dimensions or depth are invalid");
  }
  char separator = 0;
  input.get(separator);
  if (!input || !std::isspace(static_cast<unsigned char>(separator))) {
    throw std::runtime_error("Flower PPM header is not terminated");
  }
  std::vector<uint8_t> bytes(width * height * 3);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("Flower PPM pixel data is truncated");
  }

  ImageStorage image({width, height});
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      for (size_t channel = 0; channel < 3; ++channel) {
        image.planes()[channel][y * image.stride() + x] =
            SrgbToLinear(bytes[(y * width + x) * 3 + channel]);
      }
    }
  }
  return image;
}

[[nodiscard]] ImageStorage CropImage(const ImageStorage &source, size_t crop_x,
                                     size_t crop_y, OracleExtent crop_extent) {
  if (crop_extent.width == 0 || crop_extent.height == 0) {
    return source;
  }
  if (crop_x > source.extent().width - crop_extent.width ||
      crop_y > source.extent().height - crop_extent.height) {
    throw std::runtime_error("Flower crop is outside the source image");
  }
  ImageStorage cropped(crop_extent);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < crop_extent.height; ++y) {
      std::copy_n(source.planes()[channel].data() +
                      (crop_y + y) * source.stride() + crop_x,
                  crop_extent.width,
                  cropped.planes()[channel].data() + y * cropped.stride());
    }
  }
  return cropped;
}

[[nodiscard]] ImageStorage BoxBlur3x3(const ImageStorage &source) {
  ImageStorage blurred(source.extent());
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < source.extent().height; ++y) {
      for (size_t x = 0; x < source.extent().width; ++x) {
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
          const size_t source_y = static_cast<size_t>(std::clamp<ptrdiff_t>(
              static_cast<ptrdiff_t>(y) + dy, 0,
              static_cast<ptrdiff_t>(source.extent().height - 1)));
          for (int dx = -1; dx <= 1; ++dx) {
            const size_t source_x = static_cast<size_t>(std::clamp<ptrdiff_t>(
                static_cast<ptrdiff_t>(x) + dx, 0,
                static_cast<ptrdiff_t>(source.extent().width - 1)));
            sum +=
                source.planes()[channel][source_y * source.stride() + source_x];
          }
        }
        blurred.planes()[channel][y * blurred.stride() + x] = sum / 9.0f;
      }
    }
  }
  return blurred;
}

} // namespace

ImageStorage::ImageStorage(OracleExtent extent, size_t extra_stride, float fill)
    : extent_(extent), stride_(extent.width + extra_stride) {
  if (extent.width == 0 || extent.height == 0 || stride_ < extent.width ||
      extent.height > std::numeric_limits<size_t>::max() / stride_) {
    throw std::invalid_argument("Invalid Butteraugli fixture extent");
  }
  for (std::vector<float> &values : plane_) {
    values.assign(stride_ * extent.height, fill);
  }
}

ConstOracleImage3 ImageStorage::ConstView() const {
  return ConstOracleImage3{{{
      {plane_[0].data(), extent_, stride_},
      {plane_[1].data(), extent_, stride_},
      {plane_[2].data(), extent_, stride_},
  }}};
}

FixturePair MakeFixture(const FixtureSpec &spec) {
  ImageStorage reference(spec.extent);
  ImageStorage distorted(spec.extent);
  FillPattern(spec, &reference, &distorted);
  return {
      spec.name,
      std::move(reference),
      std::move(distorted),
      spec.options,
  };
}

FixturePair LoadFlowerFixture(const std::string &ppm_path, size_t crop_x,
                              size_t crop_y, OracleExtent crop_extent) {
  ImageStorage full = LoadLinearPpm(ppm_path);
  ImageStorage reference = CropImage(full, crop_x, crop_y, crop_extent);
  ImageStorage distorted = BoxBlur3x3(reference);
  return {
      crop_extent.width == 0 ? "flower_full" : "flower_crop_96x96",
      std::move(reference),
      std::move(distorted),
      {},
  };
}

std::vector<FixturePair>
BuildSyntheticDifferentialCorpus() {
  std::vector<FixturePair> corpus;
  auto add = [&](FixtureSpec spec) { corpus.push_back(MakeFixture(spec)); };
  add({"identity_texture_32x24", {32, 24}, FixtureKind::kIdentityTexture});
  for (const OracleExtent extent :
       std::array<OracleExtent, 4>{{{1, 1}, {3, 7}, {7, 3}, {8, 8}}}) {
    for (size_t channel = 0; channel < 3; ++channel) {
      add({
          "impulse_" + std::to_string(extent.width) + "x" +
              std::to_string(extent.height) + "_c" + std::to_string(channel),
          extent,
          FixtureKind::kImpulse,
          {},
          channel,
      });
    }
  }
  add({"gradient_9x13", {9, 13}, FixtureKind::kGradient});
  add({"texture_32x24", {32, 24}, FixtureKind::kTexture});
  add({"contrast_33x17", {33, 17}, FixtureKind::kContrast});
  add({"chromatic_17x29", {17, 29}, FixtureKind::kChromatic});
  add({
      "random_low_31x23",
      {31, 23},
      FixtureKind::kRandomLow,
      {},
      0,
      0x9e3779b9u,
  });
  add({
      "random_normal_13x9",
      {13, 9},
      FixtureKind::kRandomNormal,
      {},
      0,
      0x243f6a88u,
  });
  add({
      "random_wide_23x15",
      {23, 15},
      FixtureKind::kRandomWide,
      {},
      0,
      0xb7e15162u,
  });
  add({
      "texture_hf_asymmetry_1.6",
      {32, 24},
      FixtureKind::kTexture,
      {.hf_asymmetry = 1.6f},
  });
  add({
      "texture_x_multiplier_0.75",
      {32, 24},
      FixtureKind::kTexture,
      {.x_multiplier = 0.75f},
  });
  add({
      "texture_intensity_target_255",
      {32, 24},
      FixtureKind::kTexture,
      {.intensity_target = 255.0f},
  });
  return corpus;
}

std::vector<FixturePair>
BuildDifferentialCorpus(const std::string &flower_ppm_path) {
  std::vector<FixturePair> corpus = BuildSyntheticDifferentialCorpus();
  corpus.push_back(LoadFlowerFixture(flower_ppm_path, 207, 218, {96, 96}));
  return corpus;
}

bool PaddingIsPoisoned(const ImageStorage &image) {
  for (const std::vector<float> &plane : image.planes()) {
    for (size_t y = 0; y < image.extent().height; ++y) {
      for (size_t x = image.extent().width; x < image.stride(); ++x) {
        if (plane[y * image.stride() + x] != kFixturePoison) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace gjxl::butteraugli_test
