// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Compare unchanged and experimental GPU metrics on identical decoded pixels.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "codec/butteraugli.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/butteraugli.h"
#include "gpu_test_utils.h"
#include "io/pfm.h"

namespace {
void Check(gjxl::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
struct Result { double score = 0; std::vector<float> map; };
Result Evaluate(const char* library, const gjxl::Image3FBuffer& reference,
                const gjxl::Image3FBuffer& decoded) {
  const auto extent = reference.extent();
  std::unique_ptr<gjxl::GpuBackend> gpu;
  Check(gjxl::CreateMetalBackend(library, &gpu));
  std::array<gjxl::test::GuardedDevicePlane, 3> r, d;
  gjxl::test::GuardedDevicePlane map, score;
  for (size_t c=0; c<3; ++c) {
    Check(r[c].Prepare(*gpu, extent, extent.width+3+c));
    Check(d[c].Prepare(*gpu, extent, extent.width+7+c));
    r[c].SetLogical(reference.plane(c)); d[c].SetLogical(decoded.plane(c));
    Check(r[c].Upload()); Check(d[c].Upload());
  }
  Check(map.Prepare(*gpu, extent, extent.width+17));
  Check(score.Prepare(*gpu, {1,1}, 4));
  map.PoisonLogical(); score.PoisonLogical();
  Check(map.Upload()); Check(score.Upload());
  gjxl::ConstDeviceImage3View rv{{r[0].ConstView(),r[1].ConstView(),r[2].ConstView()}};
  gjxl::ConstDeviceImage3View dv{{d[0].ConstView(),d[1].ConstView(),d[2].ConstView()}};
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  Check(gjxl::PrepareDeviceButteraugli(*gpu, {rv, {}}, &prepared));
  Check(prepared->Compare({dv,map.View(),score.View()}));
  Result result;
  Check(prepared->ReadScore(&result.score));
  Check(map.Download()); Check(score.Download());
  if (!map.GuardsIntact() || !score.GuardsIntact())
    throw std::runtime_error("Output guard or padding changed");
  for (size_t c=0; c<3; ++c) {
    Check(r[c].Download()); Check(d[c].Download());
    const auto rl=r[c].Logical(), dl=d[c].Logical();
    if (!r[c].GuardsIntact() || !d[c].GuardsIntact() ||
        std::memcmp(rl.data(),reference.plane(c).data(),rl.size()*sizeof(float)) ||
        std::memcmp(dl.data(),decoded.plane(c).data(),dl.size()*sizeof(float)))
      throw std::runtime_error("Read-only input or padding changed");
  }
  result.map=map.Logical();
  for (float value:result.map) if (!std::isfinite(value) || value<0)
    throw std::runtime_error("Nonfinite or negative distance-map value");
  return result;
}
}
int main(int argc,char** argv) {
  try {
    if (argc!=5) throw std::runtime_error("usage: probe BASELINE.metallib CANDIDATE.metallib REFERENCE.pfm DECODED.pfm");
    gjxl::Image3FBuffer reference,decoded;
    Check(gjxl::io::ReadPfm(argv[3],&reference));
    Check(gjxl::io::ReadPfm(argv[4],&decoded));
    if(reference.extent()!=decoded.extent()) throw std::runtime_error("Dimensions differ");
    const auto baseline=Evaluate(argv[1],reference,decoded);
    const auto candidate=Evaluate(argv[2],reference,decoded);
    const auto extent=reference.extent();
    Result cpu;
    cpu.map.resize(extent.width*extent.height);
    Check(gjxl::ComputeButteraugliDistance(reference.const_view(),decoded.const_view(),{},
        {cpu.map.data(),extent,extent.width},&cpu.score));
    double max_delta=0,maximum=0,cpu_max_delta=0,squared=0;
    for(size_t i=0;i<baseline.map.size();++i) {
      const double delta=double(candidate.map[i])-baseline.map[i];
      max_delta=std::max(max_delta,std::abs(delta)); squared+=delta*delta;
      maximum=std::max(maximum,double(baseline.map[i]));
      cpu_max_delta=std::max(cpu_max_delta,std::abs(double(candidate.map[i])-cpu.map[i]));
    }
    const double score_budget=std::max(.002,.005*baseline.score);
    const double map_budget=std::max(.01,.01*maximum);
    std::cout<<std::setprecision(17)<<"{\"baseline_score\":"<<baseline.score
      <<",\"candidate_score\":"<<candidate.score<<",\"cpu_score\":"<<cpu.score
      <<",\"score_delta\":"<<candidate.score-baseline.score
      <<",\"map_max_abs\":"<<max_delta<<",\"map_rmse\":"<<std::sqrt(squared/baseline.map.size())
      <<",\"candidate_cpu_map_max_abs\":"<<cpu_max_delta
      <<",\"score_budget\":"<<score_budget<<",\"map_budget\":"<<map_budget
      <<",\"within_budget\":"<<((std::abs(candidate.score-baseline.score)<=score_budget&&max_delta<=map_budget)?"true":"false")
      <<",\"guards\":true,\"readonly\":true}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
