// SPDX-License-Identifier: Apache-2.0
#include "gpu/ops/ac_strategy_capture_internal.h"
#include "codec/ac_strategy_search_policy.h"
#include "graph.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string_view>
#include <vector>

namespace gjxl::frontier_experiment {
namespace {
template<class T> void Write(std::ofstream& f, const T& value) {
  f.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
}
Status Capture(Extent2D pixels, ConstPlaneF32View quant,
    const ColorCorrelationMap& cfl, AcStrategySearchOptions options,
    const ac_strategy_internal::CandidateCostTableView& table,
    const AcStrategyGrid& greedy) {
  const char* directory = std::getenv("GJXL_FRONTIER_CAPTURE_DIR");
  if (directory == nullptr || *directory == '\0') return Status::Ok();
  if (options.dense_dct32_search)
    return Status::InvalidArgument("Frontier experiment captures ordinary bank only");
  try {
    static std::atomic<unsigned> sequence{0};
    const auto prefix = std::filesystem::path(directory) /
      ("capture-" + std::to_string(sequence.fetch_add(1)));
    std::filesystem::create_directories(directory);
    if (std::filesystem::exists(prefix.string() + ".gfc"))
      return Status::InvalidArgument("Frontier capture destination already exists");
    std::vector<double> milliseconds;
    for (int i = 0; i < 9; ++i) {
      AcStrategyGrid control;
      const auto begin = std::chrono::steady_clock::now();
      Status status = ac_strategy_internal::FindAcStrategyGridFromResidentCandidateCosts(
        pixels, quant, {}, cfl, options, table, &control);
      const auto end = std::chrono::steady_clock::now();
      if (!status.ok()) return status;
      for (size_t y=0; y<table.block_extent.height; ++y)
        for (size_t x=0; x<table.block_extent.width; ++x) {
          AcStrategyCell a,b;
          if (!control.Get(x,y,&a).ok() || !greedy.Get(x,y,&b).ok() ||
              a.strategy != b.strategy || a.is_anchor != b.is_anchor)
            return Status::Internal("Frontier CPU control changed selected grid");
        }
      if (i >= 2) milliseconds.push_back(
        std::chrono::duration<double,std::milli>(end-begin).count());
    }
    std::ofstream f(prefix.string()+".gfc", std::ios::binary);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    f.write("GJFDP001",8);
    const uint32_t bw=table.block_extent.width, bh=table.block_extent.height;
    Write(f,bw); Write(f,bh); Write(f,options.butteraugli_target);
    const float multiplier=1.0f + -0.4f/(options.butteraugli_target+1.4f);
    Write(f,multiplier);
    const uint32_t count=((bw+7)/8)*((bh+7)/8);
    Write(f,count);
    for (uint32_t ty=0;ty<bh;ty+=8) for(uint32_t tx=0;tx<bw;tx+=8) {
      const uint32_t w=std::min(8u,bw-tx),h=std::min(8u,bh-ty);
      std::vector<float> raw,policy;
      for (const auto& stage: ac_strategy_internal::kCandidateStages) {
        const auto covered=GetAcStrategyInfo(stage.strategy)->covered_blocks;
        for(size_t y=0;y+covered.height<=h;y+=stage.anchor_step)
          for(size_t x=0;x+covered.width<=w;x+=stage.anchor_step) {
            const float v=table.strategy_costs[size_t(stage.strategy)][(ty+y)*bw+tx+x];
            raw.push_back(v);
            policy.push_back(stage.strategy==AcStrategyType::kDct8 ? v*multiplier:v);
          }
      }
      Write(f,w);Write(f,h);Write(f,uint32_t(raw.size()));
      f.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(float));
      f.write(reinterpret_cast<const char*>(policy.data()),policy.size()*sizeof(float));
      std::array<uint8_t,64> cells;cells.fill(255);
      for(uint32_t y=0;y<h;++y) for(uint32_t x=0;x<w;++x) {
        AcStrategyCell cell;
        const Status status=greedy.Get(tx+x,ty+y,&cell);
        if (!status.ok()) return status;
        cells[y*8+x]=(uint8_t(cell.strategy)<<1)|cell.is_anchor;
      }
      f.write(reinterpret_cast<const char*>(cells.data()),cells.size());
    }
    f.close();
    std::ofstream j(prefix.string()+".json");
    j << std::setprecision(10) << "{\"block_width\":"<<bw
      <<",\"block_height\":"<<bh<<",\"tiles\":"<<count
      <<",\"distance\":"<<options.butteraugli_target
      <<",\"dct8_multiplier\":"<<multiplier
      <<",\"cpu_greedy_ms\":[";
    for(size_t i=0;i<milliseconds.size();++i)j<<(i?",":"")<<milliseconds[i];
    j << "],\"timing_boundary\":\"host resident-cost-table selection, including grid allocation and export; excluding scoring, readback, dense-table construction and validation\"}\n";
  } catch (const std::exception& e) { return Status::Internal(e.what()); }
  return Status::Ok();
}
Status SelectDiagnostic(AcStrategySearchOptions options,
    const ac_strategy_internal::CandidateCostTableView& table,
    AcStrategyGrid* output) {
  const char* name=std::getenv("GJXL_AC_SEARCH_EXPERIMENT");
  if(name==nullptr||std::string_view(name)=="greedy")return Status::Ok();
  const bool rectangle=std::string_view(name)=="rectangle";
  if(!rectangle&&std::string_view(name)!="frontier")
    return Status::InvalidArgument("Unknown diagnostic AC solver");
  if(options.dense_dct32_search)
    return Status::InvalidArgument("Diagnostic AC solvers require ordinary candidate bank");
  try {
    // Geometry is immutable and shared between encodes on each calling thread.
    thread_local std::array<std::unique_ptr<frontier::Graph>,64> graphs;
    AcStrategyGrid result;
    Status status=AcStrategyGrid::Create(table.block_extent,&result);
    if(!status.ok())return status;
    const float multiplier=1.0f + -0.4f/(options.butteraugli_target+1.4f);
    for(unsigned ty=0;ty<table.block_extent.height;ty+=8)
      for(unsigned tx=0;tx<table.block_extent.width;tx+=8) {
        unsigned w=std::min<size_t>(8,table.block_extent.width-tx);
        unsigned h=std::min<size_t>(8,table.block_extent.height-ty);
        auto& graph=graphs[(h-1)*8+w-1];
        if(!graph)graph=std::make_unique<frontier::Graph>(w,h);
        const auto& g=*graph;
        std::vector<float> policy;policy.reserve(g.rect.size());
        for(const auto& r:g.rect) {
          float c=table.strategy_costs[r.strategy][(ty+r.y)*table.block_extent.width+tx+r.x];
          policy.push_back(r.strategy==0?c*multiplier:c);
        }
        std::vector<frontier::Wide> costs;
        const auto meta=frontier::Normalize(policy,&costs);
        if(meta.flags==2)return Status::Internal("Invalid diagnostic AC candidate cost");
        auto solution=rectangle?frontier::RectangleOracle(g,costs,meta.flags==1)
                               :frontier::Oracle(g,costs,meta.flags==1);
        frontier::Wide greedy_cost{};std::array<uint8_t,64> greedy;greedy.fill(255);
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x) {
          AcStrategyCell cell;status=output->Get(tx+x,ty+y,&cell);
          if(!status.ok())return status;
          greedy[y*8+x]=(uint8_t(cell.strategy)<<1)|cell.is_anchor;
        }
        for(unsigned c=0;c<g.rect.size();++c){auto r=g.rect[c];
          if(greedy[r.y*8+r.x]==uint8_t((r.strategy<<1)|1))greedy_cost=greedy_cost+costs[c];}
        if(!rectangle&&greedy_cost<solution.cost)
          return Status::Internal("Frontier optimum exceeds greedy cost");
        const auto& selected=solution.cost<greedy_cost?solution.grid:greedy;
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)if(selected[y*8+x]&1) {
          status=result.Set(tx+x,ty+y,AcStrategyType(selected[y*8+x]>>1));
          if(!status.ok())return status;
        }
      }
    if(!result.complete())return Status::Internal("Diagnostic AC cover is incomplete");
    *output=std::move(result);
    return Status::Ok();
  }catch(const std::exception& e){return Status::Internal(e.what());}
}
} // namespace gjxl::frontier_experiment
