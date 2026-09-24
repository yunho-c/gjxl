// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cmath>
#include <iostream>
#include <atomic>
#include <thread>
#include "cuda_sparse_resident_fixture.h"
#include "codestream/encoder_internal.h"
#include "codec/vardct_frame_view_internal.h"
#include "codec/reconstruction.h"
#include "codec/loop_filter.h"
namespace {
using namespace resident_sparse_test;
size_t evaluations=0,comparisons=0,oracle_evaluations=0,sparse_calls=0,dense_calls=0,transitions=0;
size_t Nonzeros(const VarDctEncoderFrame& frame) {
  size_t count=0;
  for(size_t g=0;g<frame.ac_group_count();++g) {
    VarDctNativeAcGroupView v;Check(frame.GetNativeAcGroup(g,&v));
    std::visit([&](const auto& group){for(const auto c:group.coefficients)for(auto x:c)count+=x!=0;},v);
  }
  return count;
}
Status NoFinal(Fixture& f,PreparedAqEvaluation& p,Output& out) {
  return p.Evaluate({.quant_field={f.field.data(),f.blocks,f.blocks.width},.quant_dc=f.quant_dc},
    {.block_distance_map={out.block.data(),f.blocks,f.blocks.width},.score=&out.score,.quantizer=&out.quantizer});
}
void MixedDcReconstruction(GpuBackend& backend) {
  size_t cases = 0;
  for (size_t side : {size_t{8}, size_t{36}}) {
    const auto mixed = gjxl_test::MakeFrame(7, 0, side);
    Fixture fixture(mixed.geometry().frame(), 0.7f);
    fixture.strategies = mixed.strategies();
    for (unsigned policy = 0; policy < 4; ++policy) {
      for (bool smoothing : {false, true}) {
      for (bool filtering : {false, true}) {
        AqEvaluationOptions options;
        options.profile.loop_filter.gaborish = filtering;
        options.profile.loop_filter.epf_options.iterations = filtering ? 2 : 0;
        options.profile.adaptive_dc_smoothing = smoothing;
        options.profile.extra_dc_precision = policy == 0 ? 0 : 1;
        options.dc_quantization = policy < 2 ? DcQuantizationMode::kRound
                                             : DcQuantizationMode::kPredictionAware;
        options.dc_prediction = policy == 3 ? VarDctDcPrediction::kWeighted
                                            : VarDctDcPrediction::kGradient;
        auto prepared = fixture.Prepare(backend, options);
        for (unsigned method : {0u, 2u}) {
          auto actual = fixture.NewOutput();
          Check(fixture.Run(*prepared, actual, method));
          Image3FBuffer opsin(fixture.padded_extent), filtered(fixture.extent),
              expected(fixture.extent);
          std::vector<float> sigma(fixture.block_count);
          Check(ReconstructQuantizedCoefficients(actual.frame, opsin.view()));
          Check(ComputeEpfInverseSigma(fixture.strategies, actual.frame.raw_quant_field(),
              actual.frame.quantizer(), {fixture.sharpness.data(), fixture.blocks, fixture.blocks.width},
              options.profile.epf_sigma, {sigma.data(), fixture.blocks, fixture.blocks.width}));
          Check(ApplyLoopFilters(opsin.cropped_view(fixture.extent), {sigma.data(), fixture.blocks, fixture.blocks.width},
              options.profile.loop_filter, filtered.view()));
          Check(OpsinToLinearRgb(filtered.cropped_view(fixture.extent),
              options.profile.intensity_target, expected.view()));
          for (size_t c = 0; c < 3; ++c)
            for (size_t i = 0; i < expected.plane(c).size(); ++i) {
              const float want = expected.plane(c)[i], got = actual.rgb.plane(c)[i];
              if (!std::isfinite(got) || std::abs(got - want) > 2.0e-4f + 2.0e-4f * std::abs(want)) {
                std::cerr << "Mixed DC reconstruction mismatch: side=" << side << " policy=" << policy
                          << " smoothing=" << smoothing << " method=" << method
                          << " filtering=" << filtering
                          << " channel=" << c << " pixel=" << i << " expected=" << want
                          << " actual=" << got << '\n';
                throw std::runtime_error("CUDA mixed DC reconstruction differs from stored-frame CPU oracle");
              }
            }
          ++cases;
        }
      }
      }
    }
  }
  std::cout << cases << " mixed-transform CUDA DC reconstructions match the CPU frame consumer.\n";
}
void CompletedLease() {
  for (auto extent : {Extent2D{17, 33}, {257, 263}}) {
    for (bool evaluate_final : {false, true}) {
      std::unique_ptr<vardct_frame_internal::CompletedVarDctFrame> lease;
      std::vector<uint8_t> expected;
      std::vector<double> scores;
      {
        std::unique_ptr<GpuBackend> gpu;
        Check(CreateCudaBackend(&gpu));
        Fixture fixture(extent, 0.0f);
        auto prepared = fixture.Prepare(*gpu);
        auto reference = fixture.NewOutput();
        Check(fixture.Run(*prepared, reference, evaluate_final ? 0 : 1));
        expected = reference.Bytes();
        const AqResidentButteraugliPolicyInput input{
          .adjusted_initial_quant_field = {fixture.field.data(), fixture.blocks, fixture.blocks.width},
          .quant_dc = fixture.quant_dc, .butteraugli_target = 1.1f,
          .lower_bound = 0.1f, .upper_bound = 10.0f, .iterations = 1,
          .evaluate_final_field = evaluate_final};
        Check(prepared->EvaluateResidentButteraugliPolicy(input,
          {.score_history = &scores, .completed_frame = &lease}));
        Require(lease != nullptr && lease->view().valid() && scores == reference.scores,
                "Completed lease or score publication is invalid");
        const auto* before = lease.get();
        const auto previous_scores = scores;
        Require(prepared->EvaluateResidentButteraugliPolicy(input,
          {.score_history = &scores, .frame = &reference.frame, .completed_frame = &lease}).code() ==
            StatusCode::kInvalidArgument && lease.get() == before && scores == previous_scores,
                "Conflicting frame outputs were not rejected atomically");
        // Reuse the producer, then destroy it, its backend and its input images.
        std::fill(fixture.field.begin(), fixture.field.end(), 0.2f);
        auto reused = fixture.NewOutput();
        Check(fixture.Run(*prepared, reused, 1));
      }
      std::vector<uint8_t> actual;
      Check(codestream_internal::EncodeVarDctCodestreamFromView(lease->view(), {}, &actual));
      Require(actual == expected, "Completed frame lease depended on producer lifetime or reuse");
    }
  }
  std::cout << "Completed CUDA leases preserve native coefficients across producer reuse/destruction.\n";
}
void Case(Fixture& f,GpuBackend& backend) {
  auto p=f.Prepare(backend),control=f.Prepare(backend);
  bool header_allocated=false,previous_sparse=false,had_frame=false;
  const size_t count=f.block_count*3*64;
  const auto initial_stats=p->memory_stats();
  const size_t header_bytes=(count/64+(count%64!=0))*12+4;
  std::atomic<bool> invalid_stats{false};
  std::jthread stats_reader([&](std::stop_token stop) {
    while(!stop.stop_requested()) {
      const auto stats=p->memory_stats();
      const size_t extra=stats.staging_bytes-initial_stats.staging_bytes;
      if((extra!=0&&extra!=header_bytes)||stats.peak_scratch_bytes!=initial_stats.peak_scratch_bytes+extra)
        invalid_stats.store(true,std::memory_order_relaxed);
      std::this_thread::yield();
    }
  });
  for(float field:{0.8f,0.0001f,16.0f,0.8f}) {
    std::fill(f.field.begin(),f.field.end(),field);
    std::vector<Output> expected;
    for(unsigned method=0;method<3;++method) {
      auto out=f.NewOutput();Check(f.Run(*control,out,method));expected.push_back(std::move(out));++oracle_evaluations;
    }
    for(unsigned method:{1u,0u,2u,1u,2u,0u}) {
      auto out=f.NewOutput();const auto before=backend.stats();Check(f.Run(*p,out,method));const auto after=backend.stats();++evaluations;
      Equal(expected[method],out);++comparisons;
      Require(gjxl_test::CheckResidentPopulation(out.frame),"Resident sparse populations differ from independent scalar recount");
      const auto info=vardct_frame_internal::GetAcStorageInfo(out.frame);
      const size_t nonzeros=Nonzeros(out.frame);
      const bool sparse=count>=size_t{3}*256*256&&nonzeros<=count/8;
      Require(info.sparse==sparse,"Resident sparse selection differs from exact density");
      Require(after.successful_allocations-before.successful_allocations==size_t{sparse&&!header_allocated},"Resident sparse header was not lazy/reused");
      header_allocated|=sparse;sparse_calls+=sparse;dense_calls+=!sparse;
      if(had_frame&&previous_sparse!=sparse)++transitions;
      previous_sparse=sparse;had_frame=true;
      // Test-only materialization: production keeps only the native owner.
      gjxl_test::PopulationAssembly dense(out.frame);VarDctEncoderFrame dense_frame;
      Check(vardct_frame_internal::AssembleVarDctEncoderFrame(dense.Input(nullptr),&dense_frame));
      std::vector<uint8_t> bytes;Check(EncodeVarDctCodestream(dense_frame,{},&bytes));
      Require(bytes==out.Bytes(),"Resident sparse serializer differs from dense materialization");
    }
    auto expected_no_final=f.NewOutput(),actual_no_final=f.NewOutput();
    Check(NoFinal(f,*control,expected_no_final));++oracle_evaluations;
    Check(NoFinal(f,*p,actual_no_final));++evaluations;Equal(expected_no_final,actual_no_final);++comparisons;
  }
  stats_reader.request_stop();stats_reader.join();
  Require(!invalid_stats.load(std::memory_order_relaxed),"Concurrent sparse memory statistics are inconsistent");
}
}
int main(){try {
  std::unique_ptr<GpuBackend> backend;const auto status=CreateCudaBackend(&backend);
  if(status.code()==StatusCode::kUnavailable){std::cout<<status.message()<<'\n';return 77;}Check(status);
  MixedDcReconstruction(*backend);
  CompletedLease();
  size_t cases=0;
  for(unsigned shape=0;shape<6;++shape)for(float amplitude:{0.0f,0.01f,1.0f,16.0f}) {
    const std::array<Extent2D,5> sizes{{{1,1},{17,33},{65,67},{257,263},{513,519}}};
    const auto mixed=gjxl_test::MakeFrame(7,0,36);Fixture f(shape<5?sizes[shape]:mixed.geometry().frame(),amplitude);
    if(shape==5)f.strategies=mixed.strategies();Case(f,*backend);++cases;
  }
  Require(sparse_calls&&dense_calls&&transitions,"Resident sparse transition coverage is incomplete");
  std::cout<<"CUDA SPARSE RESIDENT PASS cases="<<cases<<" evaluations="<<evaluations<<" comparisons="<<comparisons<<" oracle_evaluations="<<oracle_evaluations<<" sparse_calls="<<sparse_calls<<" dense_calls="<<dense_calls<<" transitions="<<transitions<<'\n'<<std::flush;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n'<<std::flush;return 1;}}
