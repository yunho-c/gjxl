// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#include "core/ac_strategy.h"
#include "gpu/cuda/cuda_ac_group_kernels.h"
#include "gpu/cuda/cuda_direct_ac_pack_kernels.h"

namespace {
using namespace gjxl;
using namespace gjxl::cuda_internal;
constexpr std::array strategies{
  AcStrategyType::kDct8, AcStrategyType::kDct16x16, AcStrategyType::kDct32x32,
  AcStrategyType::kDct16x8, AcStrategyType::kDct8x16,
  AcStrategyType::kDct32x16, AcStrategyType::kDct16x32};
constexpr uint32_t guard = 0xa193c7e5;

void Require(bool okay, const char* message) {
  if (!okay) throw std::runtime_error(message);
}
void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void Check(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
struct Device {
  void* data = nullptr;
  explicit Device(size_t size) { Check(cudaMalloc(&data, size)); }
  ~Device() { if (data) cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  template<typename T> T* As() const { return static_cast<T*>(data); }
};
struct Record {
  size_t x, y, tag, group, used;
  size_t source = 0, stride = 0;
};
struct Fixture {
  Extent2D blocks;
  size_t count;
  std::vector<CudaAqAnchor> anchors{7, {9876,5432}};
  std::vector<uint64_t> offsets = std::vector<uint64_t>(7, UINT64_MAX);
  std::array<CudaAqExactBatch,7> batches{};
  std::vector<Record> records;
  std::vector<size_t> sizes, bases;
  Fixture(Extent2D dimensions, size_t pattern) : blocks(dimensions),
      count(3*blocks.width*blocks.height*64) {
    AcStrategyGrid grid;
    Check(AcStrategyGrid::Create(blocks,&grid));
    for (size_t y=0;y<blocks.height;++y) for (size_t x=0;x<blocks.width;++x) {
      if (grid.occupied(x,y)) continue;
      auto strategy=strategies[pattern<7?pattern:(x+3*y)%7];
      const auto covered=GetAcStrategyInfo(strategy)->covered_blocks;
      if (x/32!=(x+covered.width-1)/32 || y/32!=(y+covered.height-1)/32 ||
          !grid.Set(x,y,strategy).ok()) Check(grid.Set(x,y,AcStrategyType::kDct8));
    }
    const size_t gw=(blocks.width+31)/32, groups=gw*((blocks.height+31)/32);
    std::vector<size_t> used(groups,0);
    sizes.resize(groups); bases.resize(groups);
    size_t packed=0;
    for (size_t g=0;g<groups;++g) {
      sizes[g]=std::min<size_t>(32,blocks.width-(g%gw)*32)*
        std::min<size_t>(32,blocks.height-(g/gw)*32)*64;
      bases[g]=packed; packed+=3*sizes[g];
    }
    Require(packed==count,"Invalid fixture capacity");
    std::array<std::vector<size_t>,7> grouped;
    Check(grid.ForEachAnchor([&](size_t x,size_t y,AcStrategyType strategy) {
      const size_t tag=std::find(strategies.begin(),strategies.end(),strategy)-strategies.begin();
      const size_t group=(y/32)*gw+x/32;
      grouped[tag].push_back(records.size());
      records.push_back({x,y,tag,group,used[group]});
      used[group]+=GetAcStrategyInfo(strategy)->coefficient_count();
      return Status::Ok();
    }));
    Require(used==sizes,"Incomplete fixture");
    size_t source=0;
    for (size_t tag=0;tag<7;++tag) {
      const auto* info=GetAcStrategyInfo(strategies[tag]);
      auto& b=batches[tag];
      b={static_cast<uint32_t>(anchors.size()),static_cast<uint32_t>(grouped[tag].size()),
         static_cast<uint32_t>(source),static_cast<uint32_t>(info->coefficient_count()),
         static_cast<uint32_t>(info->pixel_extent().width),static_cast<uint32_t>(info->pixel_extent().height),
         static_cast<uint32_t>(info->covered_blocks.width),static_cast<uint32_t>(info->covered_blocks.height)};
      for (size_t i=0;i<grouped[tag].size();++i) {
        auto& r=records[grouped[tag][i]];
        r.source=source+i*b.coefficient_count; r.stride=grouped[tag].size()*b.coefficient_count;
        anchors.push_back({static_cast<uint32_t>(r.x),static_cast<uint32_t>(r.y)});
        offsets.push_back(bases[r.group]+r.used);
      }
      source+=3*grouped[tag].size()*b.coefficient_count;
    }
    Require(source==count,"Invalid fixture source");
  }
  std::vector<int32_t> Packed(std::span<const int32_t> source) const {
    std::vector<int32_t> result(count);
    for (const auto& r:records) for (size_t c=0;c<3;++c)
      std::copy_n(source.data()+r.source+c*r.stride,
        GetAcStrategyInfo(strategies[r.tag])->coefficient_count(),
        result.data()+bases[r.group]+c*sizes[r.group]+r.used);
    return result;
  }
};

size_t cases=0;
void Case(Extent2D blocks,size_t strategy) {
  const Fixture f(blocks,strategy);
  Device da(f.anchors.size()*sizeof(CudaAqAnchor)), di(f.offsets.size()*8),
    ds((f.count+48)*4), dd((f.count+48)*4), db(f.count+64), dw(2*f.count+64), df(12);
  Check(cudaMemcpy(da.data,f.anchors.data(),f.anchors.size()*sizeof(CudaAqAnchor),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(di.data,f.offsets.data(),f.offsets.size()*8,cudaMemcpyHostToDevice));
  for (size_t pattern=0;pattern<6;++pattern) {
    std::vector<int32_t> source(f.count+48,std::bit_cast<int32_t>(guard));
    uint32_t expected_flags=0;
    for (size_t i=0;i<f.count;++i) {
      int32_t value=0;
      if (pattern==1) value=i&1?-128:127;
      if (pattern==2) value=i&1?-32768:32767;
      if (pattern==3) value=i&1?INT32_MIN:INT32_MAX;
      if (pattern==4) value=i+1==f.count?128:-128;
      if (pattern==5) value=i+1==f.count?-32769:32767;
      source[i+17]=value;
      expected_flags|=(value<-128||value>127)?1u:0u;
      expected_flags|=(value<-32768||value>32767)?2u:0u;
    }
    const auto packed=f.Packed(std::span(source).subspan(17,f.count));
    std::vector<uint32_t> bytes(f.count/4+16,guard),words(f.count/2+16,guard);
    std::vector<int32_t> dense(f.count+48,std::bit_cast<int32_t>(guard));
    std::array<uint32_t,3> flags{guard,0,guard};
    Check(cudaMemcpy(ds.data,source.data(),source.size()*4,cudaMemcpyHostToDevice));
    Check(cudaMemcpy(db.data,bytes.data(),bytes.size()*4,cudaMemcpyHostToDevice));
    Check(cudaMemcpy(dw.data,words.data(),words.size()*4,cudaMemcpyHostToDevice));
    Check(cudaMemcpy(df.data,flags.data(),12,cudaMemcpyHostToDevice));
    for (const auto& batch:f.batches)
      Check(LaunchCudaPackCompactAcGroups(da.As<CudaAqAnchor>(),di.As<uint64_t>(),ds.As<int32_t>()+17,
        db.As<uint32_t>()+8,dw.As<uint32_t>()+8,df.As<uint32_t>()+1,batch,
        static_cast<uint32_t>(blocks.width),static_cast<uint32_t>(blocks.height),nullptr));
    Check(cudaDeviceSynchronize());
    Check(cudaMemcpy(bytes.data(),db.data,bytes.size()*4,cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(words.data(),dw.data,words.size()*4,cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(flags.data(),df.data,12,cudaMemcpyDeviceToHost));
    Require(flags==std::array<uint32_t,3>{guard,expected_flags,guard},"Flags/guards differ");
    for (size_t i=0;i<8;++i)
      Require(bytes[i]==guard && bytes[bytes.size()-1-i]==guard &&
              words[i]==guard && words[words.size()-1-i]==guard,"Payload guard differs");
    for (size_t i=0;i<f.count;++i) {
      uint16_t word;
      std::memcpy(&word,reinterpret_cast<uint8_t*>(words.data()+8)+2*i,2);
      Require(reinterpret_cast<uint8_t*>(bytes.data()+8)[i]==(uint32_t(packed[i])&255u) &&
              word==(uint32_t(packed[i])&65535u),"Packed values differ");
    }
    // Exercise the untouched source through the real dense fallback packer.
    Check(cudaMemcpy(dd.data,dense.data(),dense.size()*4,cudaMemcpyHostToDevice));
    for (const auto& batch:f.batches)
      Check(LaunchCudaPackAcGroups(da.As<CudaAqAnchor>(),di.As<uint64_t>(),ds.As<int32_t>()+17,
        dd.As<int32_t>()+17,batch,static_cast<uint32_t>(blocks.width),static_cast<uint32_t>(blocks.height),nullptr));
    Check(cudaDeviceSynchronize());
    Check(cudaMemcpy(dense.data(),dd.data,dense.size()*4,cudaMemcpyDeviceToHost));
    Require(std::equal(packed.begin(),packed.end(),dense.begin()+17),"Dense fallback differs");
    for (size_t i=0;i<dense.size();++i)
      if (i<17||i>=17+f.count) Require(dense[i]==std::bit_cast<int32_t>(guard),"Dense guard differs");
    auto returned=source;
    Check(cudaMemcpy(returned.data(),ds.data,returned.size()*4,cudaMemcpyDeviceToHost));
    Require(returned==source,"Source changed");
    ++cases;
  }
  auto anchors=f.anchors;auto offsets=f.offsets;
  Check(cudaMemcpy(anchors.data(),da.data,anchors.size()*sizeof(CudaAqAnchor),cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(offsets.data(),di.data,offsets.size()*8,cudaMemcpyDeviceToHost));
  Require(std::memcmp(anchors.data(),f.anchors.data(),anchors.size()*sizeof(CudaAqAnchor))==0 &&
          offsets==f.offsets,"Metadata changed");
}
}
int main(int argc,char** argv) {
  if (cudaFree(nullptr)!=cudaSuccess) return 77;
  try {
    const bool sanitizer=argc==2&&std::string_view(argv[1])=="--sanitizer";
    CudaAqExactBatch empty;
    Check(LaunchCudaPackCompactAcGroups(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,empty,0,0,nullptr));
    empty.anchor_count=1;
    Require(LaunchCudaPackCompactAcGroups(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,empty,0,0,nullptr)==cudaErrorInvalidValue,
      "Invalid dispatch accepted");
    for (Extent2D blocks:{Extent2D{1,1},{3,5},{31,29},{32,32},{33,33},{64,67},{240,135},{480,270}}) {
      if (sanitizer&&blocks.width>33) continue;
      for (size_t strategy=0;strategy<8;++strategy) Case(blocks,strategy);
    }
    std::cout<<"CUDA direct AC PASS cases="<<cases<<'\n'<<std::flush;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n'<<std::flush;return 1;
  }
}
