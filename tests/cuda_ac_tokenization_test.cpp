// SPDX-License-Identifier: Apache-2.0
// Independent public-tokenizer oracle for the private CUDA experiment.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "codestream/ac_tokenization_provider_internal.h"
#include "gpu/cuda/cuda_ac_tokenization_kernels.h"
#include "quantized_frame_fixture.h"

using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::cuda_internal;
using gjxl_test::Check;
void Require(bool v, const char* message) {
  if (!v) throw std::runtime_error(message);
}
void Cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <class T>
class Device {
 public:
  explicit Device(const std::vector<T>& source) : count_(source.size()) {
    Cuda(cudaMalloc(reinterpret_cast<void**>(&base_),
                    (count_ + 64) * sizeof(T)));
    Cuda(cudaMemset(base_, 0xa5, (count_ + 64) * sizeof(T)));
    Upload(source);
  }
  ~Device() {
    if (base_) cudaFree(base_);
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  T* get() const { return base_ + 32; }
  void Upload(const std::vector<T>& source) {
    Require(source.size() == count_, "Upload size changed");
    Cuda(cudaMemcpy(get(), source.data(), count_ * sizeof(T),
                    cudaMemcpyHostToDevice));
  }
  std::vector<T> Read() const {
    std::vector<T> data(count_ + 64);
    Cuda(cudaMemcpy(data.data(), base_, data.size() * sizeof(T),
                    cudaMemcpyDeviceToHost));
    const auto* bytes = reinterpret_cast<const unsigned char*>(data.data());
    for (size_t i = 0; i < 32 * sizeof(T); ++i)
      Require(bytes[i] == 0xa5 && bytes[(count_ + 32) * sizeof(T) + i] == 0xa5,
              "CUDA token buffer guard changed");
    return {data.begin() + 32, data.end() - 32};
  }

 private:
  T* base_ = nullptr;
  size_t count_;
};

int main(int argc, char** argv) {
  try {
    const bool smoke = argc == 2 && std::strcmp(argv[1], "--smoke") == 0;
    Require(argc == 1 || smoke, "Unknown arguments");
    Cuda(cudaSetDevice(0));
    Require(LaunchCudaAcTokenization({}, 0, 0, false, nullptr) == cudaSuccess,
            "Empty launch rejected");
    Require(LaunchCudaAcTokenization({}, 1, 1, false, nullptr) ==
                cudaErrorInvalidValue,
            "Null nonempty launch accepted");
    Require(LaunchCudaAcTokenization({}, 0, 1, false, nullptr) ==
                cudaErrorInvalidValue,
            "Inconsistent counts accepted");
    SimpleAcNaturalOrders natural;
    Check(PrepareSimpleAcNaturalOrders(&natural));
    auto qf = DefaultSimpleBlockContextMap();
    qf.qf_thresholds = {16, 64};
    qf.num_contexts = 6;
    qf.context_map.resize(3 * 13 * 3);
    for (size_t i = 0; i < qf.context_map.size(); ++i)
      qf.context_map[i] = (i / 3 + i % 3) % 6;
    const std::array maps = {DefaultSimpleBlockContextMap(),
                             JxlDefaultSimpleBlockContextMap(),
                             TwoChannelSimpleBlockContextMap(), qf};
    size_t cases = 0, groups_checked = 0, retries = 0;
    for (size_t strategy = 0; strategy <= gjxl_test::kStrategies.size();
         ++strategy) {
      if (smoke && strategy != 0 && strategy != 7) continue;
      for (size_t pattern = 0; pattern < 8; ++pattern) {
        if (smoke && pattern != 0 && pattern != 5 && pattern != 7) continue;
        const auto frame = gjxl_test::MakeFrame(strategy, pattern);
        const uint32_t stride = 65536 + 17 * static_cast<uint32_t>(pattern);
        std::vector<int32_t> packed(64 + frame.ac_group_count() * 3 * stride,
                                    0x12345678);
        for (size_t g = 0; g < frame.ac_group_count(); ++g) {
          VarDctAcGroupView view;
          Check(frame.GetAcGroup(g, &view));
          for (size_t c = 0; c < 3; ++c)
            std::copy_n(view.coefficients[c].begin(),
                        view.used_coefficient_count,
                        packed.begin() + 64 + g * 3 * stride + c * stride);
        }
        Device<int32_t> coefficients(packed);
        SimpleCoefficientOrders custom;
        Check(ComputeSimpleCoefficientOrders(frame, &custom));
        for (const auto& order :
             std::array{SimpleCoefficientOrders{}, custom}) {
          std::vector<CudaTokenStrategy> strategies(kAcStrategyCount);
          std::vector<uint32_t> scans;
          for (size_t i = 0; i < kAcStrategyCount; ++i) {
            if (natural.orders[i].empty()) continue;
            const auto& info =
                *GetAcStrategyInfo(static_cast<AcStrategyType>(i));
            auto& s = strategies[i];
            s.count = static_cast<uint32_t>(info.coefficient_count());
            s.width = static_cast<uint32_t>(info.covered_blocks.width);
            s.height = static_cast<uint32_t>(info.covered_blocks.height);
            s.covered = s.width * s.height;
            s.log2covered = std::countr_zero(s.covered);
            const size_t family = kSimpleStrategyOrder[i];
            uint32_t offsets[3];
            for (size_t c = 0; c < 3; ++c) {
              const auto& scan = order.used_order_mask & (uint16_t{1} << family)
                                     ? order.orders[family][c]
                                     : natural.orders[i];
              offsets[c] = static_cast<uint32_t>(scans.size());
              scans.insert(scans.end(), scan.begin(), scan.end());
            }
            s.order0 = offsets[0];
            s.order1 = offsets[1];
            s.order2 = offsets[2];
          }
          std::vector<CudaTokenAnchor> anchors;
          std::vector<CudaTokenGroup> groups(frame.ac_group_count());
          uint32_t map_size = 0;
          for (uint32_t g = 0; g < groups.size(); ++g) {
            VarDctAcGroupView view;
            Check(frame.GetAcGroup(g, &view));
            auto& group = groups[g];
            group.first = static_cast<uint32_t>(anchors.size());
            group.width = static_cast<uint32_t>(view.block_extent.width);
            group.height = static_cast<uint32_t>(view.block_extent.height);
            group.map = map_size;
            map_size += 3 * group.width * group.height;
            uint32_t source = 64 + g * 3 * stride;
            for (uint32_t y = 0; y < group.height; ++y)
              for (uint32_t x = 0; x < group.width; ++x) {
                AcStrategyCell cell;
                Check(frame.strategies().Get(view.block_x + x, view.block_y + y,
                                             &cell));
                if (!cell.is_anchor) continue;
                const auto s = static_cast<uint32_t>(cell.strategy);
                const auto quant = frame.raw_quant_field().Row(
                    view.block_y + y)[view.block_x + x];
                anchors.push_back({source, g,
                                   x | (y << 5) | (s << 10) |
                                       (static_cast<uint32_t>(quant - 1) << 15),
                                   stride});
                source += strategies[s].count;
                ++group.anchors;
              }
          }
          for (const auto& map : maps) {
            std::vector<SimpleAcGroupTokenStream> reference;
            Check(TokenizeSimpleAcGroups(frame, order, map, &reference));
            Storage<PreparedFixedAnsCluster> expected(map.ac_context_count());
            uint32_t token_count = 0;
            for (const auto& group : reference) {
              token_count += static_cast<uint32_t>(group.tokens.size());
              for (const auto token : group.tokens) {
                HybridUintToken encoded;
                Check(EncodeHybridUint(token.value, kDefaultHybridUintConfig,
                                       &encoded));
                auto& pop = expected[token.context];
                ++pop.counts[encoded.symbol];
                ++pop.token_count;
                pop.extra_bits += encoded.extra_bit_count;
                pop.maximum_symbol =
                    std::max(pop.maximum_symbol, encoded.symbol);
              }
            }
            for (bool collect : {false, true})
              for (bool scalar : {false, true}) {
                const bool retry = cases % 17 == 0;
                std::vector<uint32_t> control{
                    map.num_contexts,
                    static_cast<uint32_t>(map.qf_thresholds.size() + 1), 1,
                    static_cast<uint32_t>(anchors.size() * 3)};
                control.insert(control.end(), map.qf_thresholds.begin(),
                               map.qf_thresholds.end());
                control.insert(control.end(), map.context_map.begin(),
                               map.context_map.end());
                const size_t end = control.size();
                control.insert(control.end(),
                               {uint32_t(collect), retry ? 1 : token_count, 0,
                                1, uint32_t(scalar)});
                Device<CudaTokenAnchor> da(anchors);
                Device<CudaTokenGroup> dg(groups);
                Device<CudaTokenStrategy> ds(strategies);
                Device<uint32_t> order_device(scans), dc(control);
                Device<CudaTokenMetadata> dm(
                    std::vector<CudaTokenMetadata>(anchors.size() * 3));
                Device<uint8_t> dmap(std::vector<uint8_t>(map_size, 0xa5));
                Device<uint32_t> values(
                    std::vector<uint32_t>(token_count, 0xa5a5a5a5));
                Device<uint16_t> contexts(
                    std::vector<uint16_t>(token_count, 0xa5a5));
                Device<uint32_t> histogram(
                    std::vector<uint32_t>(map.ac_context_count() * 128));
                CudaTokenPointers p{coefficients.get(),
                                    da.get(),
                                    dg.get(),
                                    ds.get(),
                                    order_device.get(),
                                    dm.get(),
                                    dmap.get(),
                                    values.get(),
                                    contexts.get(),
                                    histogram.get(),
                                    dc.get()};
                Cuda(LaunchCudaAcTokenization(
                    p, control[3], static_cast<uint32_t>(groups.size()), scalar,
                    nullptr));
                Cuda(cudaDeviceSynchronize());
                const auto counts = dc.Read();
                Require(counts[end + 2] == token_count,
                        "Compact total differs");
                if (retry) {
                  for (auto v : values.Read())
                    Require(v == 0xa5a5a5a5, "Overflow published values");
                  for (auto c : contexts.Read())
                    Require(c == 0xa5a5, "Overflow published contexts");
                  for (auto h : histogram.Read())
                    Require(h == 0, "Overflow published population");
                  control = counts;
                  control[end + 1] = token_count;
                  dc.Upload(control);
                  Cuda(LaunchCudaAcTokenEmit(p, control[3], scalar, nullptr));
                  Cuda(cudaDeviceSynchronize());
                  ++retries;
                }
                const auto actual_groups = dg.Read();
                const auto actual_values = values.Read();
                const auto actual_contexts = contexts.Read();
                uint32_t offset = 0;
                for (size_t g = 0; g < groups.size(); ++g) {
                  Require(actual_groups[g].output == offset,
                          "Group prefix differs");
                  Require(actual_groups[g].tokens == reference[g].tokens.size(),
                          "Group count differs");
                  for (size_t i = 0; i < reference[g].tokens.size(); ++i) {
                    const auto token = reference[g].tokens[i];
                    Require(actual_values[offset + i] == token.value &&
                                actual_contexts[offset + i] == token.context,
                            "Token/context differs");
                  }
                  offset += actual_groups[g].tokens;
                  ++groups_checked;
                }
                const auto hist = histogram.Read();
                Storage<PreparedFixedAnsCluster> populations(
                    map.ac_context_count());
                for (size_t c = 0; c < populations.size(); ++c)
                  for (uint32_t s = 0; s < 128; ++s) {
                    const uint64_t count = hist[c * 128 + s];
                    auto& pop = populations[c];
                    pop.counts[s] = count;
                    pop.token_count += count;
                    if (count) pop.maximum_symbol = s;
                    if (s >= 16)
                      pop.extra_bits += count * (2 + ((s - 16) >> 2));
                  }
                Require(
                    populations == (collect ? expected
                                            : Storage<PreparedFixedAnsCluster>(
                                                  map.ac_context_count())),
                    "Fixed populations differ");
                da.Read();
                ds.Read();
                order_device.Read();
                dm.Read();
                dmap.Read();
                ++cases;
              }
          }
        }
        Require(coefficients.Read() == packed,
                "Read-only coefficients changed");
      }
    }
    Require(cases == (smoke ? 192 : 2048) && groups_checked == 4 * cases,
            "Coverage incomplete");
    std::cout << "Verified " << cases << " CUDA token cases, " << groups_checked
              << " groups, " << retries
              << " guarded retries, three launch validation cases.\n" << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
