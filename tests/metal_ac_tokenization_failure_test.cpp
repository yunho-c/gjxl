// SPDX-License-Identifier: Apache-2.0
#include "codestream/ac_tokenization_provider_internal.h"
#include "gpu/metal/metal_backend_internal.h"
#include "quantized_frame_fixture.h"
#include <iostream>
#include <stdexcept>
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
    auto &metal = static_cast<metal_internal::MetalBackend &>(*gpu);
    auto frame = gjxl_test::MakeFrame(0, 5, 4);
    auto view = vardct_frame_internal::BorrowFrame(frame);
    SimpleAcNaturalOrders natural;
    Check(PrepareSimpleAcNaturalOrders(&natural));
    const auto map = DefaultSimpleBlockContextMap();
    VarDctAcGroupView group;
    Check(frame.GetAcGroup(0, &group));
    std::vector<int32_t> coefficients(196608, 0x13579);
    for (size_t c = 0; c < 3; ++c)
      std::copy_n(group.coefficients[c].begin(), group.used_coefficient_count,
                  coefficients.begin() + c * 65536);
    std::unique_ptr<DeviceBuffer> buffer;
    Check(gpu->Allocate(coefficients.size() * 4, &buffer));
    Check(gpu->CopyHostToDevice(*buffer, coefficients.data(),
                                coefficients.size() * 4));
    std::vector<SimpleAcGroupTokenStream> expected;
    Check(TokenizeSimpleAcGroups(frame, {}, map, &expected));
    auto verify = [&](const auto &streams) {
      Require(streams.size() == 1 &&
                  streams[0].size() == expected[0].tokens.size(),
              "Recovery token count differs");
      for (size_t i = 0; i < streams[0].size(); ++i)
        Require(streams[0][i] == expected[0].tokens[i],
                "Recovery tokens differ");
    };
    size_t cases = 0;
    setenv("GJXL_EXPERIMENT_TOKEN_CACHE", "0", 1);
    unsetenv("GJXL_EXPERIMENT_TOKEN_KERNEL_PROFILE");
    for (const char *mode : {"0", "1", "2"}) {
      setenv("GJXL_EXPERIMENT_TOKEN_COMPACT", mode, 1);
      setenv("GJXL_EXPERIMENT_TOKEN_CAPACITY", "1",
             1); // Force the guarded retry in mode 2.
      for (int fault = 0; fault < 5; ++fault) {
        if (mode[0] == '0' && fault >= 3)
          continue;
        std::unique_ptr<AcTokenizationProvider> provider;
        Check(CreateMetalAcTokenizationProvider(*gpu, *buffer, 0, &provider));
        if (fault == 0)
          metal.ArmNextAllocationFailureForTest();
        if (fault == 1)
          metal.ArmNextSubmissionFailureForTest(true, false);
        if (fault == 2)
          metal.ArmNextSubmissionFailureForTest(false, true);
        auto status = provider->Begin(view, {}, natural, map, true);
        uint32_t sentinel = 444;
        uint16_t sentinel_context = 7;
        Storage<EntropyTokenStreamView> streams{EntropyTokenStreamView::Split(
            {&sentinel, 1}, {&sentinel_context, 1})};
        Storage<PreparedFixedAnsCluster> populations(1);
        populations[0].token_count = 777;
        if (fault < 2)
          Require(!status.ok(), "Begin injection succeeded unexpectedly");
        else {
          Check(status);
          if (fault == 3)
            metal.ArmNextSubmissionFailureForTest(true, false);
          if (fault == 4)
            metal.ArmNextSubmissionFailureForTest(false, true);
          status = provider->Finish(&streams, &populations);
          Require(!status.ok(), "Finish injection succeeded unexpectedly");
        }
        Require(streams.size() == 1 && streams[0].size() == 1 &&
                    streams[0][0].value == 444 && populations.size() == 1 &&
                    populations[0].token_count == 777,
                "Failed GPU tokenizer published partial output");
        provider.reset();
        Check(CreateMetalAcTokenizationProvider(*gpu, *buffer, 0, &provider));
        Check(provider->Begin(view, {}, natural, map, true));
        Check(provider->Finish(&streams, &populations));
        verify(streams);
        Require(!provider->Finish(&streams, &populations).ok(),
                "Repeated Finish accepted");
        ++cases;
      }
    }
    std::unique_ptr<DeviceBuffer> tiny;
    Check(gpu->Allocate(4, &tiny));
    {
      std::unique_ptr<AcTokenizationProvider> provider;
      Check(CreateMetalAcTokenizationProvider(*gpu, *tiny, 0, &provider));
      const auto before = gpu->stats().committed_submissions;
      Require(!provider->Begin(view, {}, natural, map, true).ok() &&
                  gpu->stats().committed_submissions == before,
              "Truncated coefficients reached GPU");
    }
    setenv("GJXL_EXPERIMENT_TOKEN_CACHE", "1", 1);
    unsetenv("GJXL_EXPERIMENT_TOKEN_CAPACITY");
    // Two overlapping leases must never reuse live token storage; trimming idle
    // cache during consumption must not invalidate either stream.
    std::array<std::unique_ptr<AcTokenizationProvider>, 2> providers;
    std::array<Storage<EntropyTokenStreamView>, 2> streams;
    std::array<Storage<PreparedFixedAnsCluster>, 2> populations;
    for (auto &p : providers) {
      Check(CreateMetalAcTokenizationProvider(*gpu, *buffer, 0, &p));
      Check(p->Begin(view, {}, natural, map, true));
    }
    for (int i = 1; i >= 0; --i) {
      Check(providers[i]->Finish(&streams[i], &populations[i]));
      verify(streams[i]);
    }
    Check(gpu->TrimPreparationCache());
    for (auto &s : streams)
      verify(s);
    providers = {};
    Check(gpu->TrimPreparationCache());
    std::cout << "Verified " << cases
              << " allocation/submission/completion failures and recovery, "
                 "guarded compact retries, no partial publication, bounds, "
                 "overlapping leases and cache trim.\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
