// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <iostream>
#include <atomic>
#include <thread>
#include "cuda_sparse_resident_fixture.h"
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
  size_t cases=0;
  for(unsigned shape=0;shape<6;++shape)for(float amplitude:{0.0f,0.01f,1.0f,16.0f}) {
    const std::array<Extent2D,5> sizes{{{1,1},{17,33},{65,67},{257,263},{513,519}}};
    const auto mixed=gjxl_test::MakeFrame(7,0,36);Fixture f(shape<5?sizes[shape]:mixed.geometry().frame(),amplitude);
    if(shape==5)f.strategies=mixed.strategies();Case(f,*backend);++cases;
  }
  Require(sparse_calls&&dense_calls&&transitions,"Resident sparse transition coverage is incomplete");
  std::cout<<"CUDA SPARSE RESIDENT PASS cases="<<cases<<" evaluations="<<evaluations<<" comparisons="<<comparisons<<" oracle_evaluations="<<oracle_evaluations<<" sparse_calls="<<sparse_calls<<" dense_calls="<<dense_calls<<" transitions="<<transitions<<'\n'<<std::flush;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n'<<std::flush;return 1;}}
