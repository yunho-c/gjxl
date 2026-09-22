// SPDX-License-Identifier: Apache-2.0
// Exact GPU tokens and fixed populations against the independent public
// tokenizer.
#include "codestream/ac_tokenization_provider_internal.h"
#include "gpu/metal/metal_backend.h"
#include "quantized_frame_fixture.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace gjxl;
using namespace gjxl::codestream_internal;
using gjxl_test::Check;
static void Require(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
int main() {
  try {
    std::unique_ptr<GpuBackend> gpu;
    Check(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
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
    size_t cases = 0, groups = 0;
    for (size_t strategy = 0; strategy <= gjxl_test::kStrategies.size();
         ++strategy) {
      for (size_t pattern = 0; pattern < 8; ++pattern) {
        const auto frame = gjxl_test::MakeFrame(strategy, pattern);
        // Native resident layout has fixed channel/group strides. Copy only the
        // used spans, with a nonzero buffer offset and poisoned unused
        // elements.
        std::vector<int32_t> packed(64 + frame.ac_group_count() * 196608,
                                    0x12345678);
        for (size_t g = 0; g < frame.ac_group_count(); ++g) {
          VarDctAcGroupView view;
          Check(frame.GetAcGroup(g, &view));
          for (size_t c = 0; c < 3; ++c)
            std::copy_n(view.coefficients[c].begin(),
                        view.used_coefficient_count,
                        packed.begin() + 64 + g * 196608 + c * 65536);
        }
        std::unique_ptr<DeviceBuffer> buffer;
        Check(gpu->Allocate(packed.size() * sizeof(int32_t), &buffer));
        Check(gpu->CopyHostToDevice(*buffer, packed.data(),
                                    packed.size() * sizeof(int32_t)));
        SimpleCoefficientOrders custom;
        Check(ComputeSimpleCoefficientOrders(frame, &custom));
        for (const auto &order : std::array{SimpleCoefficientOrders{}, custom})
          for (const auto &map : maps) {
            std::vector<SimpleAcGroupTokenStream> reference;
            Check(TokenizeSimpleAcGroups(frame, order, map, &reference));
            Storage<PreparedFixedAnsCluster> expected(map.ac_context_count());
            for (const auto &stream : reference)
              for (const auto t : stream.tokens) {
                HybridUintToken encoded;
                Check(EncodeHybridUint(t.value, kDefaultHybridUintConfig,
                                       &encoded));
                auto &p = expected[t.context];
                ++p.counts[encoded.symbol];
                ++p.token_count;
                p.extra_bits += encoded.extra_bit_count;
                p.maximum_symbol = std::max(p.maximum_symbol, encoded.symbol);
              }
            for (bool collect : {false, true, true, false, false}) {
              const char *shards[] = {"1", "2", "4", "8"};
              setenv("GJXL_EXPERIMENT_TOKEN_SHARDS", shards[(cases / 2) % 4],
                     1);
              std::unique_ptr<AcTokenizationProvider> provider;
              Check(CreateMetalAcTokenizationProvider(*gpu, *buffer, 256,
                                                      &provider));
              Check(provider->Begin(vardct_frame_internal::BorrowFrame(frame),
                                    order, natural, map, collect));
              Storage<EntropyTokenStreamView> actual;
              Storage<PreparedFixedAnsCluster> populations;
              Check(provider->Finish(&actual, &populations));
              Require(actual.size() == reference.size(),
                      "GPU group count differs");
              for (size_t g = 0; g < reference.size(); ++g) {
                Require(actual[g].size() == reference[g].tokens.size(),
                        "GPU token count differs");
                for (size_t i = 0; i < actual[g].size(); ++i) {
                  auto a = actual[g][i], b = reference[g].tokens[i];
                  if (a.value != b.value || a.context != b.context) {
                    std::cerr << "case=" << cases << " strategy=" << strategy
                              << " pattern=" << pattern << " group=" << g
                              << " token=" << i << " actual=" << a.context
                              << ',' << a.value << " expected=" << b.context
                              << ',' << b.value << '\n';
                    throw std::runtime_error(
                        "GPU token differs from independent public tokenizer");
                  }
                }
                ++groups;
              }
              Require(collect ? populations == expected : populations.empty(),
                      "GPU fixed populations differ");
              ++cases;
            }
          }
      }
    }
    Require(cases == 2560 && groups == 10240, "GPU coverage incomplete");
    std::cout << "Verified " << cases << " GPU submissions and " << groups
              << " groups: exact tokens, contexts, histograms, extra bits and "
                 "maximum symbols.\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
