// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstring>
#include <iostream>
#include <vector>

#include "gpu/cuda/cuda_ac_tokenization.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "quantized_frame_fixture.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::cuda_internal;
using gjxl_test::Check;
void Require(bool value, const char* text) {
  if (!value) throw std::runtime_error(text);
}
std::vector<int32_t> Pack(const VarDctEncoderFrame& frame) {
  std::vector<int32_t> result(16, 0x12345678);
  for (size_t g = 0; g < frame.ac_group_count(); ++g) {
    VarDctNativeAcGroupView view;
    Check(frame.GetNativeAcGroup(g, &view));
    std::visit(
        [&](const auto& native) {
          for (const auto channel : native.coefficients)
            for (size_t i = 0; i < native.used_coefficient_count; ++i)
              result.push_back(channel[i]);
        },
        view);
  }
  result.insert(result.end(), 16, 0x12345678);
  return result;
}
void Failures(GpuBackend& backend, const VarDctEncoderFrame& frame,
              const DeviceBuffer& coefficients,
              const SimpleAcNaturalOrders& natural) {
  const auto view = vardct_frame_internal::BorrowFrame(frame);
  const auto map = DefaultSimpleBlockContextMap();
  std::unique_ptr<AcTokenizationProvider> provider;
  const EntropyToken sentinel{7, 42};
  const auto reset = [&] {
    provider.reset();
    Check(
        CreateCudaAcTokenizationProvider(backend, coefficients, 64, &provider));
  };
  const auto finish_failure = [&](bool inject_host_failure = false) {
    Storage<EntropyTokenStreamView> streams{
        EntropyTokenStreamView::Interleaved({&sentinel, 1})};
    Storage<PreparedFixedAnsCluster> populations(1);
    populations[0].token_count = 77;
    // Construct caller-owned sentinels before arming the serializer allocator.
    if (inject_host_failure)
      resource_budget_internal::
          ArmManagedHostClassAllocationFailureAfterForTest(
              resource_budget_internal::ResourceClass::kSerializer, 0);
    Require(!provider->Finish(&streams, &populations).ok(),
            "Injected Finish failure accepted");
    if (inject_host_failure)
      Require(!resource_budget_internal::
                  ManagedHostAllocationFailurePendingForTest(),
              "Finish did not consume the host allocation failure");
    Require(streams.size() == 1 && streams[0][0] == sentinel &&
                populations.size() == 1 && populations[0].token_count == 77,
            "Failed Finish changed caller outputs");
  };
  reset();
  finish_failure();
  auto* cuda = dynamic_cast<CudaBackend*>(&backend);
  Require(cuda != nullptr, "Backend type");
  reset();
  Check(backend.TrimPreparationCache());
  cuda->ArmNextAllocationFailureForTest();
  Require(!provider->Begin(view, {}, natural, map, true).ok(),
          "Injected device allocation accepted");
  reset();
  resource_budget_internal::ArmManagedHostClassAllocationFailureAfterForTest(
      resource_budget_internal::ResourceClass::kSerializer, 0);
  Require(!provider->Begin(view, {}, natural, map, true).ok(),
          "Injected host allocation accepted");
  resource_budget_internal::DisarmManagedHostAllocationFailureForTest();
  reset();
  Check(ArmNextCudaSubmissionFailureForTest(backend, true, false));
  Require(!provider->Begin(view, {}, natural, map, true).ok(),
          "Injected submission failure accepted");
  reset();
  Check(ArmNextCudaSubmissionFailureForTest(backend, false, true));
  Check(provider->Begin(view, {}, natural, map, true));
  finish_failure();
  reset();
  Check(provider->Begin(view, {}, natural, map, true));
  finish_failure(true);
  resource_budget_internal::DisarmManagedHostAllocationFailureForTest();
  reset();
  Check(provider->Begin(view, {}, natural, map, true));
  // Dropping an in-flight provider must wait before freeing metadata/output.
  provider.reset();
  std::unique_ptr<DeviceBuffer> short_buffer;
  Check(backend.Allocate(4, &short_buffer));
  Check(CreateCudaAcTokenizationProvider(backend, *short_buffer, 0, &provider));
  Require(!provider->Begin(view, {}, natural, map, true).ok(),
          "Truncated input accepted");
  reset();
  std::unique_ptr<GpuBackend> other;
  Check(CreateCudaBackend(&other));
  auto* before = provider.get();
  Require(!CreateCudaAcTokenizationProvider(*other, coefficients, 64, &provider)
                  .ok() &&
              provider.get() == before,
          "Foreign coefficient backend accepted or changed output");
  reset();
  Check(provider->Begin(view, {}, natural, map, true));
  Storage<EntropyTokenStreamView> streams;
  Storage<PreparedFixedAnsCluster> populations;
  Check(provider->Finish(&streams, &populations));
  Require(!provider->Begin(view, {}, natural, map, true).ok() &&
              !provider->Finish(&streams, &populations).ok(),
          "Repeated provider use accepted");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool smoke = argc == 2 && std::strcmp(argv[1], "--smoke") == 0;
    Require(argc == 1 || smoke, "Unknown test argument");
    std::unique_ptr<GpuBackend> backend;
    Check(CreateCudaBackend(&backend));
    SimpleAcNaturalOrders natural;
    Check(PrepareSimpleAcNaturalOrders(&natural));
    auto qf = DefaultSimpleBlockContextMap();
    qf.qf_thresholds = {16, 64};
    qf.num_contexts = 6;
    qf.context_map.resize(3 * 13 * 3);
    for (size_t i = 0; i < qf.context_map.size(); ++i)
      qf.context_map[i] = (i / 3 + i % 3) % 6;
    const std::array maps{DefaultSimpleBlockContextMap(),
                          JxlDefaultSimpleBlockContextMap(),
                          TwoChannelSimpleBlockContextMap(), qf};
    size_t cases = 0, groups = 0;
    for (size_t strategy = 0; strategy < 8; ++strategy) {
      if (smoke && strategy != 0 && strategy != 7) continue;
      for (size_t pattern : {size_t{0}, size_t{1}, size_t{5}, size_t{7}}) {
        if (smoke && pattern > 1) continue;
        auto frame = gjxl_test::MakeFrame(strategy, pattern, smoke ? 8 : 36);
        const auto packed = Pack(frame);
        std::unique_ptr<DeviceBuffer> coefficients;
        Check(backend->Allocate(packed.size() * 4, &coefficients));
        Check(backend->CopyHostToDevice(*coefficients, packed.data(),
                                        packed.size() * 4, 0));
        SimpleCoefficientOrders custom;
        Check(ComputeSimpleCoefficientOrders(frame, &custom));
        for (size_t ordering = 0; ordering < (smoke ? 1u : 2u); ++ordering) {
          const auto& order = ordering ? custom : SimpleCoefficientOrders{};
          for (size_t mapping = 0; mapping < (smoke ? 1u : maps.size());
               ++mapping) {
            const auto& map = maps[mapping];
            std::vector<SimpleAcGroupTokenStream> expected;
            Check(TokenizeSimpleAcGroups(frame, order, map, &expected));
            Storage<PreparedFixedAnsCluster> expected_populations(
                map.ac_context_count());
            for (const auto& group : expected)
              for (const auto token : group.tokens) {
                HybridUintToken encoded;
                Check(EncodeHybridUint(token.value, kDefaultHybridUintConfig,
                                       &encoded));
                auto& p = expected_populations[token.context];
                ++p.counts[encoded.symbol];
                ++p.token_count;
                p.extra_bits += encoded.extra_bit_count;
                p.maximum_symbol = std::max(p.maximum_symbol, encoded.symbol);
              }
            for (bool collect : {false, true}) {
              std::unique_ptr<AcTokenizationProvider> provider;
              Check(CreateCudaAcTokenizationProvider(*backend, *coefficients,
                                                     64, &provider));
              const auto begins = cuda_token_provider_begin_count;
              Check(provider->Begin(vardct_frame_internal::BorrowFrame(frame),
                                    order, natural, map, collect));
              Storage<EntropyTokenStreamView> streams;
              Storage<PreparedFixedAnsCluster> populations;
              Check(provider->Finish(&streams, &populations));
              Require(cuda_token_provider_begin_count == begins + 1,
                      "Provider did not submit");
              Require(streams.size() == expected.size(),
                      "Token group count differs");
              for (size_t g = 0; g < streams.size(); ++g) {
                Require(streams[g].valid() &&
                            streams[g].size() == expected[g].tokens.size(),
                        "Token stream shape differs");
                for (size_t i = 0; i < streams[g].size(); ++i)
                  Require(streams[g][i] == expected[g].tokens[i],
                          "Token value/context differs");
                ++groups;
              }
              Require(collect ? populations == expected_populations
                              : populations.empty(),
                      "Integer populations differ");
              ++cases;
            }
          }
        }
        std::vector<int32_t> readback(packed.size());
        Check(backend->CopyDeviceToHost(*coefficients, readback.data(),
                                        readback.size() * 4, 0));
        Require(readback == packed, "Input/guards modified");
        if (strategy == 7 && pattern == 1)
          Failures(*backend, frame, *coefficients, natural);
      }
    }
    Require(cuda_token_provider_retry_count > 0,
            "No compact output retry exercised");
    std::cout << "Verified " << cases << " CUDA provider cases, " << groups
              << " exact groups, failure/recovery and destruction; retries="
              << cuda_token_provider_retry_count << ".\n"
              << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n' << std::flush;
    return 1;
  }
}
