// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace gjxl::benchmark_internal {

/// Small dependency-free SHA-256 implementation used only for benchmark
/// artifacts and deterministic completed-frame fingerprints.
class Sha256 {
public:
  void Update(std::span<const uint8_t> input) {
    total_bytes_ += input.size();
    while (!input.empty()) {
      const size_t available = block_.size() - block_size_;
      const size_t copied = input.size() < available ? input.size() : available;
      for (size_t index = 0; index < copied; ++index) {
        block_[block_size_ + index] = input[index];
      }
      block_size_ += copied;
      input = input.subspan(copied);
      if (block_size_ == block_.size()) {
        Transform(block_);
        block_size_ = 0;
      }
    }
  }

  void Update(std::string_view input) {
    Update({reinterpret_cast<const uint8_t*>(input.data()), input.size()});
  }

  [[nodiscard]] std::array<uint8_t, 32> Digest() const {
    Sha256 copy = *this;
    return copy.Finalize();
  }

private:
  static constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
  };

  static constexpr uint32_t RotateRight(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32 - count));
  }

  void Transform(const std::array<uint8_t, 64>& block) {
    std::array<uint32_t, 64> words{};
    for (size_t index = 0; index < 16; ++index) {
      const size_t offset = index * 4;
      words[index] =
        (static_cast<uint32_t>(block[offset]) << 24) |
        (static_cast<uint32_t>(block[offset + 1]) << 16) |
        (static_cast<uint32_t>(block[offset + 2]) << 8) |
        static_cast<uint32_t>(block[offset + 3]);
    }
    for (size_t index = 16; index < words.size(); ++index) {
      const uint32_t s0 = RotateRight(words[index - 15], 7) ^
        RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3);
      const uint32_t s1 = RotateRight(words[index - 2], 17) ^
        RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (size_t index = 0; index < words.size(); ++index) {
      const uint32_t upper_sigma1 =
        RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t temporary1 = h + upper_sigma1 + choose +
        kRoundConstants[index] + words[index];
      const uint32_t upper_sigma0 =
        RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temporary2 = upper_sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint8_t, 32> Finalize() {
    const uint64_t bit_count = total_bytes_ * 8;
    block_[block_size_++] = 0x80;
    if (block_size_ > 56) {
      while (block_size_ < block_.size()) block_[block_size_++] = 0;
      Transform(block_);
      block_size_ = 0;
    }
    while (block_size_ < 56) block_[block_size_++] = 0;
    for (size_t index = 0; index < 8; ++index) {
      block_[56 + index] = static_cast<uint8_t>(
        bit_count >> (56 - 8 * index));
    }
    Transform(block_);

    std::array<uint8_t, 32> digest{};
    for (size_t index = 0; index < state_.size(); ++index) {
      digest[index * 4] = static_cast<uint8_t>(state_[index] >> 24);
      digest[index * 4 + 1] = static_cast<uint8_t>(state_[index] >> 16);
      digest[index * 4 + 2] = static_cast<uint8_t>(state_[index] >> 8);
      digest[index * 4 + 3] = static_cast<uint8_t>(state_[index]);
    }
    return digest;
  }

  std::array<uint32_t, 8> state_ = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
  };
  std::array<uint8_t, 64> block_{};
  size_t block_size_ = 0;
  uint64_t total_bytes_ = 0;
};

inline std::string Sha256Hex(std::span<const uint8_t> input) {
  constexpr char kHex[] = "0123456789abcdef";
  Sha256 hash;
  hash.Update(input);
  const std::array<uint8_t, 32> digest = hash.Digest();
  std::string result(64, '0');
  for (size_t index = 0; index < digest.size(); ++index) {
    result[index * 2] = kHex[digest[index] >> 4];
    result[index * 2 + 1] = kHex[digest[index] & 0x0f];
  }
  return result;
}

}  // namespace gjxl::benchmark_internal
