// SPDX-License-Identifier: Apache-2.0
#include "graph.h"
#include "codec/ac_strategy_search_internal.h"
#include "codec/chroma_from_luma_internal.h"
#include "../../benchmarks/metal_probe.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>

using namespace frontier;
using gjxl::benchmark::Buffer;
using gjxl::benchmark::Kernel;
using gjxl::benchmark::Check;
using Clock=std::chrono::steady_clock;
constexpr auto Guard=gjxl::benchmark::kGuard;
struct Tile {std::vector<float> raw,policy;std::array<uint8_t,64> greedy{};bool has_greedy=false;};
struct Batch {
  Graph g;RectangleGraph rg;Params p{};std::vector<Tile> tiles;
  std::vector<Meta> expected_meta;
  std::vector<Solution> expected;
  std::vector<Solution> rectangle;
  std::vector<Wide> normalized;
  std::vector<std::unique_ptr<Buffer>> b;
  // nodes,edges,levels,offsets,rect,raw,policy,normalized,meta,result,grid,
  // split choices,wide values,wide choices
  Batch(unsigned w,unsigned h):g(w,h),rg(g){}
  void prepare(MTL::Device* device,float multiplier,bool discount) {
    p={uint32_t(tiles.size()),uint32_t(g.rect.size()),uint32_t(g.node.size()),
       g.slot_count,g.root,g.terminal,g.w,g.h,multiplier,uint32_t(discount),0,32};
    const size_t sizes[]={g.node.size()*sizeof(Node),g.edge.size()*sizeof(Edge),
      g.level.size()*2,sizeof(g.offsets),g.rect.size()*sizeof(frontier::Rect),
      tiles.size()*g.rect.size()*4,tiles.size()*g.rect.size()*4,
      tiles.size()*g.rect.size()*8,tiles.size()*sizeof(Meta),tiles.size()*sizeof(Wide),
      tiles.size()*64,tiles.size()*g.node.size(),32*g.slot_count*sizeof(Wide),32*g.node.size(),
      rg.node.size()*sizeof(RectangleNode),rg.edge.size()*sizeof(RectangleEdge),sizeof(rg.offsets),
      32*rg.node.size()*sizeof(Wide),32*rg.node.size()*2,7*64*2};
    for(auto n:sizes)b.push_back(std::make_unique<Buffer>(device,n));
    auto copy=[&](unsigned i,const auto& v){std::memcpy(b[i]->data(),v.data(),v.size()*sizeof(v[0]));};
    copy(0,g.node);copy(1,g.edge);copy(2,g.level);copy(3,g.offsets);copy(4,g.rect);
    copy(14,rg.node);copy(15,rg.edge);copy(16,rg.offsets);
    std::fill_n(b[19]->as<uint16_t>(),7*64,65535);
    for(unsigned c=0;c<g.rect.size();++c){auto r=g.rect[c];
      for(unsigned s=0;s<7;++s)if(uint8_t(gjxl::ac_strategy_internal::kCandidateStages[s].strategy)==r.strategy)
        b[19]->as<uint16_t>()[s*64+r.y*8+r.x]=c;}
    for(unsigned t=0;t<tiles.size();++t) {
      Check(tiles[t].raw.size()==g.rect.size()&&tiles[t].policy.size()==g.rect.size(),"candidate count");
      std::memcpy(b[5]->as<float>()+t*g.rect.size(),tiles[t].raw.data(),g.rect.size()*4);
      std::vector<Wide> cost;auto m=Normalize(tiles[t].policy,&cost);expected_meta.push_back(m);
      normalized.insert(normalized.end(),cost.begin(),cost.end());
      expected.push_back(m.flags==2?Solution{}:Oracle(g,cost,m.flags==1));
      rectangle.push_back(m.flags==2?Solution{}:RectangleOracle(g,cost,m.flags==1));
      if(m.flags!=2) {
        if(!tiles[t].has_greedy) {
          // Actual production CPU policy, not a second transcription of it.
          std::array<std::vector<float>,gjxl::kAcStrategyCount> storage;
          gjxl::ac_strategy_internal::CandidateCostTableView table{.block_extent={g.w,g.h}};
          for(unsigned s=0;s<storage.size();++s){storage[s].assign(g.w*g.h,0);table.strategy_costs[s]=storage[s];}
          for(unsigned c=0;c<g.rect.size();++c){auto r=g.rect[c];storage[r.strategy][r.y*g.w+r.x]=tiles[t].policy[c];}
          std::vector<float> quant(g.w*g.h,1);int8_t zero=0;gjxl::ColorCorrelationMap cfl;
          Check(gjxl::chroma_from_luma_internal::CreateColorCorrelationMap(
            {&zero,{1,1},1},{&zero,{1,1},1},&cfl).ok(),"create CPU oracle CfL");
          gjxl::AcStrategyGrid grid;
          const auto cpu_status=gjxl::ac_strategy_internal::FindAcStrategyGridFromResidentCandidateCosts(
            {g.w*8,g.h*8},{quant.data(),{g.w,g.h},g.w},{},cfl,
            {.butteraugli_target=std::numeric_limits<float>::max()},table,&grid);
          Check(cpu_status.ok(),"CPU greedy oracle shape="+std::to_string(g.w)+"x"+std::to_string(g.h)+
            " tile="+std::to_string(t)+" "+std::string(cpu_status.message()));
          tiles[t].greedy.fill(255);
          for(unsigned y=0;y<g.h;++y)for(unsigned x=0;x<g.w;++x) {
            gjxl::AcStrategyCell cell;Check(grid.Get(x,y,&cell).ok(),"CPU oracle cell");
            tiles[t].greedy[y*8+x]=(uint8_t(cell.strategy)<<1)|cell.is_anchor;
          }
          tiles[t].has_greedy=true;
        }
        Check(!(rectangle.back().cost<expected.back().cost),"rectangle beats unrestricted optimum");
        for(const auto* solution:{&expected.back(),&rectangle.back()}) {
          std::array<unsigned,64> covered{};Wide sum{};
          for(unsigned c=0;c<g.rect.size();++c) {
            auto r=g.rect[c];
            if(solution->grid[r.y*8+r.x]!=uint8_t((r.strategy<<1)|1))continue;
            sum=sum+cost[c];
            for(unsigned y=0;y<r.h;++y)for(unsigned x=0;x<r.w;++x) {
              unsigned i=(r.y+y)*8+r.x+x;++covered[i];
              Check(solution->grid[i]==uint8_t((r.strategy<<1)|(!x&&!y)),"invalid cover cell");
            }
          }
          Check(sum==solution->cost,"cover cost mismatch");
          for(unsigned y=0;y<g.h;++y)for(unsigned x=0;x<g.w;++x)
            Check(covered[y*8+x]==1,"incomplete or overlapping cover");
        }
      }
    }
  }
  void bind(MTL::ComputeCommandEncoder* e,unsigned buffer,unsigned index) {
    e->setBuffer(b[buffer]->object.get(),Guard,index);
  }
  void validate(bool is_rectangle=false,bool is_greedy=false,bool hybrid=false)const {
    for(const auto& x:b)x->guards();
    for(unsigned t=0;t<tiles.size();++t) {
      auto m=b[8]->as<Meta>()[t];
      Check(m.flags==expected_meta[t].flags&&m.shift==expected_meta[t].shift,
        "normalization metadata tile="+std::to_string(t)+" shape="+std::to_string(g.w)+"x"+std::to_string(g.h));
      for(unsigned c=0;c<g.rect.size();++c) {
        unsigned i=t*g.rect.size()+c;
        Check(b[6]->as<uint32_t>()[i]==std::bit_cast<uint32_t>(tiles[t].policy[c]),"policy bit mismatch");
        if(!m.flags)Check(b[7]->as<uint64_t>()[i]==normalized[i].limb[0],"integer conversion mismatch");
      }
      if(m.flags==2)continue;
      if(is_greedy) {
        if(tiles[t].has_greedy&&std::memcmp(b[10]->as<uint8_t>()+64*t,tiles[t].greedy.data(),64)!=0) {
          std::cerr<<"greedy mismatch cells (expected/device):";
          for(unsigned i=0;i<64;++i)std::cerr<<" "<<unsigned(tiles[t].greedy[i])<<"/"<<unsigned(b[10]->as<uint8_t>()[64*t+i]);
          std::cerr<<"\n";
        }
        if(tiles[t].has_greedy)Check(std::memcmp(b[10]->as<uint8_t>()+64*t,tiles[t].greedy.data(),64)==0,
          "greedy CPU map mismatch shape="+std::to_string(g.w)+"x"+std::to_string(g.h)+" tile="+std::to_string(t));
        continue;
      }
      const auto& oracle=is_rectangle?rectangle[t]:expected[t];
      Wide greedy_cost{};
      if(hybrid)for(unsigned c=0;c<g.rect.size();++c){auto r=g.rect[c];
        if(tiles[t].greedy[r.y*8+r.x]==uint8_t((r.strategy<<1)|1))greedy_cost=greedy_cost+normalized[t*g.rect.size()+c];}
      const bool keep_greedy=hybrid&&!(oracle.cost<greedy_cost);
      Check(b[9]->as<Wide>()[t]==(keep_greedy?greedy_cost:oracle.cost),"exact optimum mismatch");
      Check(std::memcmp(b[10]->as<uint8_t>()+64*t,(keep_greedy?tiles[t].greedy:oracle.grid).data(),64)==0,"traceback mismatch");
    }
  }
};
struct Timing {double gpu_ms,wall_ms;};
struct Engine {
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::CommandQueue> queue;
  Kernel prepare,solve,trace,wide,rectangle,rectangle_wide,greedy;
  Engine():device(NS::TransferPtr(MTL::CreateSystemDefaultDevice())),
    queue(NS::TransferPtr(device->newCommandQueue())),
    prepare(device.get(),FRONTIER_METALLIB_PATH,"fdp_prepare"),
    solve(device.get(),FRONTIER_METALLIB_PATH,"fdp_solve"),
    trace(device.get(),FRONTIER_METALLIB_PATH,"fdp_trace"),
    wide(device.get(),FRONTIER_METALLIB_PATH,"fdp_wide"),
    rectangle(device.get(),FRONTIER_METALLIB_PATH,"rectangle_solve"),
    rectangle_wide(device.get(),FRONTIER_METALLIB_PATH,"rectangle_wide"),
    greedy(device.get(),FRONTIER_METALLIB_PATH,"greedy_solve") {
    Check(solve.pipeline->staticThreadgroupMemoryLength()<=device->maxThreadgroupMemoryLength(),"threadgroup memory exceeds device");
  }
  void encode_greedy(MTL::CommandBuffer* command,Batch& b,unsigned threads,bool hybrid=false) {
    auto encoder=NS::RetainPtr(command->computeCommandEncoder());auto e=encoder.get();
    e->setComputePipelineState(greedy.pipeline.get());
    unsigned indices[]={6,19,8,10};for(unsigned i=0;i<4;++i)b.bind(e,indices[i],i);
    auto p=b.p;p.split=hybrid;e->setBytes(&p,sizeof(p),4);b.bind(e,9,5);
    e->dispatchThreads(MTL::Size(b.p.tiles,1,1),MTL::Size(threads,1,1));e->endEncoding();
  }
  void encode_rectangle(MTL::CommandBuffer* command,Batch& b,unsigned threads,bool is_wide) {
    auto encoder=NS::RetainPtr(command->computeCommandEncoder());auto e=encoder.get();
    auto p=b.p;p.nodes=b.rg.node.size();p.root=b.rg.root;
    const auto& k=is_wide?rectangle_wide:rectangle;
    Check(threads<=k.pipeline->maxTotalThreadsPerThreadgroup(),"rectangle thread limit");
    e->setComputePipelineState(k.pipeline.get());
    if(is_wide) {
      unsigned indices[]={14,15,4,6,8,9,10,17,18};for(unsigned i=0;i<9;++i)b.bind(e,indices[i],i);
      e->setBytes(&p,sizeof(p),9);
      e->dispatchThreads(MTL::Size(p.workers,1,1),MTL::Size(32,1,1));
    } else {
      unsigned indices[]={14,15,16,4,7,8,9,10};for(unsigned i=0;i<8;++i)b.bind(e,indices[i],i);
      e->setBytes(&p,sizeof(p),8);
      e->dispatchThreadgroups(MTL::Size(p.tiles,1,1),MTL::Size(threads,1,1));
    }
    e->endEncoding();
  }
  void encode(MTL::CommandBuffer* command,Batch& b,unsigned threads,unsigned stage,bool split) {
    if(stage==0)threads=std::min(threads,256u);
    auto encoder=NS::RetainPtr(command->computeCommandEncoder());auto e=encoder.get();b.p.split=split;
    const Kernel* k=stage==0?&prepare:stage==1?&solve:stage==2?&trace:&wide;
    Check(threads<=k->pipeline->maxTotalThreadsPerThreadgroup(),"pipeline thread limit");
    e->setComputePipelineState(k->pipeline.get());
    if(stage==0) {
      unsigned indices[]={5,4,6,7,8};for(unsigned i=0;i<5;++i)b.bind(e,indices[i],i);
      e->setBytes(&b.p,sizeof(b.p),5);
      e->dispatchThreadgroups(MTL::Size(b.p.tiles,1,1),MTL::Size(threads,1,1));
    } else if(stage==1) {
      unsigned indices[]={0,1,2,3,4,7,8,9,10,11};for(unsigned i=0;i<10;++i)b.bind(e,indices[i],i);
      e->setBytes(&b.p,sizeof(b.p),10);
      e->dispatchThreadgroups(MTL::Size(b.p.tiles,1,1),MTL::Size(threads,1,1));
    } else if(stage==2) {
      unsigned indices[]={0,1,4,8,11,10};for(unsigned i=0;i<6;++i)b.bind(e,indices[i],i);
      e->setBytes(&b.p,sizeof(b.p),6);
      e->dispatchThreads(MTL::Size(b.p.tiles,1,1),MTL::Size(64,1,1));
    } else {
      unsigned indices[]={0,1,2,3,4,6,8,9,10,12,13};for(unsigned i=0;i<11;++i)b.bind(e,indices[i],i);
      e->setBytes(&b.p,sizeof(b.p),11);
      e->dispatchThreads(MTL::Size(b.p.workers,1,1),MTL::Size(32,1,1));
    }
    e->endEncoding();
  }
  // mode: 0=complete fused DP/trace, 1=prepare, 2=DP saving decisions,
  // 3=trace saved decisions, 4=wide fallback, 5=complete split diagnostics.
  Timing run(std::vector<std::unique_ptr<Batch>>& batches,unsigned threads,unsigned mode) {
    const auto start=Clock::now();auto command=NS::RetainPtr(queue->commandBuffer());
    for(auto& b:batches) {
      if(mode==7){encode(command.get(),*b,threads,0,false);encode_greedy(command.get(),*b,threads);continue;}
      if(mode==6||mode==8) {
        encode(command.get(),*b,threads,0,false);
        encode_rectangle(command.get(),*b,threads,false);
        encode_rectangle(command.get(),*b,32,true);
        if(mode==8)encode_greedy(command.get(),*b,threads,true);
        continue;
      }
      if(mode==0||mode==1||mode==5)encode(command.get(),*b,threads,0,false);
      if(mode==0||mode==2||mode==5)encode(command.get(),*b,threads,1,mode!=0);
      if(mode==3||mode==5)encode(command.get(),*b,threads,2,true);
      if(mode==0||mode==4||mode==5)encode(command.get(),*b,threads,3,false);
    }
    command->commit();command->waitUntilCompleted();
    const auto stop=Clock::now();
    Check(command->status()==MTL::CommandBufferStatusCompleted,"Metal command failed");
    return {(command->GPUEndTime()-command->GPUStartTime())*1000,
      std::chrono::duration<double,std::milli>(stop-start).count()};
  }
};
template<class T>T Read(std::ifstream& f){T v;f.read(reinterpret_cast<char*>(&v),sizeof(T));return v;}
std::vector<std::unique_ptr<Batch>> Load(const std::string& path,float* multiplier,uint32_t* bw,uint32_t* bh) {
  std::ifstream f(path,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);
  char magic[8];f.read(magic,8);Check(std::memcmp(magic,"GJFDP001",8)==0,"capture magic");
  *bw=Read<uint32_t>(f);*bh=Read<uint32_t>(f);(void)Read<float>(f);*multiplier=Read<float>(f);
  auto count=Read<uint32_t>(f);Check(count>0&&count<1000000,"tile count limit");
  std::map<std::pair<unsigned,unsigned>,std::unique_ptr<Batch>> groups;
  for(unsigned t=0;t<count;++t) {
    unsigned w=Read<uint32_t>(f),h=Read<uint32_t>(f),nc=Read<uint32_t>(f);
    auto& b=groups[{w,h}];if(!b)b=std::make_unique<Batch>(w,h);
    Check(nc==b->g.rect.size(),"capture bank mismatch");Tile tile;
    tile.raw.resize(nc);tile.policy.resize(nc);tile.has_greedy=true;
    f.read(reinterpret_cast<char*>(tile.raw.data()),nc*4);
    f.read(reinterpret_cast<char*>(tile.policy.data()),nc*4);
    f.read(reinterpret_cast<char*>(tile.greedy.data()),64);b->tiles.push_back(std::move(tile));
  }
  Check(f.peek()==std::char_traits<char>::eof(),"unexpected capture trailer");
  std::vector<std::unique_ptr<Batch>> out;for(auto& [shape,b]:groups)out.push_back(std::move(b));return out;
}
std::vector<std::unique_ptr<Batch>> Synthetic(float multiplier=1) {
  std::mt19937 rng(20260917);std::vector<std::unique_ptr<Batch>> out;
  for(unsigned h=1;h<=8;++h)for(unsigned w=1;w<=8;++w) {
    auto b=std::make_unique<Batch>(w,h);
    for(unsigned test=0;test<44;++test) {
      Tile tile;
      for(unsigned c=0;c<b->g.rect.size();++c) {
        float v=0;auto r=b->g.rect[c];
        if(test==1)v=float(r.w*r.h);
        if(test==2)v=float(1+rng()%100000);
        if(test==3)v=std::bit_cast<float>(0x41000000u+uint32_t(rng()%4));
        if(test==4)v=std::bit_cast<float>(c==0?1u:0x7f7fffffu);
        if(test==5)v=std::bit_cast<float>(0x7f7fffffu);
        if(test==6)v=std::bit_cast<float>(1u+uint32_t(rng()%0x7fffff));
        if(test==7)v=c==0?std::numeric_limits<float>::quiet_NaN():1;
        if(test==8)v=c==0?-1.f:1.f;
        if(test==9)v=c==0?std::numeric_limits<float>::infinity():1;
        if(test==10)v=-0.f;
        if(test==11)v=std::bit_cast<float>((1u+uint32_t(rng()%254))<<23|uint32_t(rng()%0x800000));
        if(test>=12)v=test%2?std::bit_cast<float>((1u+uint32_t(rng()%254))<<23|uint32_t(rng()%0x800000))
                            :float(1+rng()%1000000);
        tile.raw.push_back(v);tile.policy.push_back(r.strategy==0?v*multiplier:v);
      }
      b->tiles.push_back(std::move(tile));
    }
    out.push_back(std::move(b));
  }
  return out;
}
long double Approximate(const Wide& v,unsigned shift) {
  long double n=0;for(int i=4;i>=0;--i)n=std::ldexp(n,64)+v.limb[i];return std::ldexp(n,int(shift)-149);
}
int main(int argc,char** argv) {
  auto pool=NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  try {
    std::string capture,output;bool self=false,discount_test=false;unsigned samples=11,warmups=3;
    for(int i=1;i<argc;++i) {
      std::string a=argv[i];if(a=="--self-test"){self=true;continue;}
      if(a=="--self-test-discount"){self=true;discount_test=true;continue;}
      Check(i+1<argc,"missing argument");std::string v=argv[++i];
      if(a=="--capture")capture=v;else if(a=="--out")output=v;
      else if(a=="--samples")samples=std::stoul(v);else if(a=="--warmups")warmups=std::stoul(v);
      else throw std::runtime_error("unknown argument "+a);
    }
    Check(self!=!capture.empty(),"select exactly one of --self-test or --capture");
    Check(samples>0&&samples<=100&&warmups<=20,"bounded sample counts");
    Engine engine;float multiplier=discount_test?0.77777779f:1;uint32_t bw=0,bh=0;
    auto batches=self?Synthetic(multiplier):Load(capture,&multiplier,&bw,&bh);
    const auto oracle_start=Clock::now();
    for(auto& b:batches)b->prepare(engine.device.get(),multiplier,!self||discount_test);
    const double oracle_setup=std::chrono::duration<double,std::milli>(Clock::now()-oracle_start).count();
    unsigned tiles=0,fallback=0,invalid=0,improved=0,ties=0,changed=0,rectangle_gap=0;
    long double exact_total=0,greedy_total=0,rectangle_total=0;
    for(auto& b:batches)for(unsigned t=0;t<b->tiles.size();++t) {
      ++tiles;auto m=b->expected_meta[t];fallback+=m.flags==1;invalid+=m.flags==2;
      if(m.flags==2||!b->tiles[t].has_greedy)continue;
      Wide greedy{};
      for(unsigned c=0;c<b->g.rect.size();++c) {
        auto r=b->g.rect[c];
        if(b->tiles[t].greedy[r.y*8+r.x]==uint8_t((r.strategy<<1)|1))greedy=greedy+b->normalized[t*b->g.rect.size()+c];
      }
      Check(!(greedy<b->expected[t].cost),"exact objective exceeds existing selector");
      improved+=b->expected[t].cost<greedy;ties+=b->expected[t].cost==greedy;
      changed+=b->tiles[t].greedy!=b->expected[t].grid;
      exact_total+=Approximate(b->expected[t].cost,m.shift);greedy_total+=Approximate(greedy,m.shift);
      const auto rect_cost=b->rectangle[t].cost<greedy?b->rectangle[t].cost:greedy;
      rectangle_gap+=b->expected[t].cost<rect_cost;
      rectangle_total+=Approximate(rect_cost,m.shift);
    }
    std::ofstream file;if(!output.empty()){file.open(output);file.exceptions(std::ios::badbit|std::ios::failbit);}
    std::ostream& out=output.empty()?std::cout:file;
    out<<std::setprecision(10)<<"{\"device\":\""<<engine.device->name()->utf8String()<<"\",\"capture\":\""<<capture
      <<"\",\"self_test\":"<<(self?"true":"false")<<",\"block_width\":"<<bw<<",\"block_height\":"<<bh
      <<",\"dct8_multiplier\":"<<multiplier<<",\"discount_test\":"<<(discount_test?"true":"false")
      <<",\"tiles\":"<<tiles<<",\"shapes\":"<<batches.size()<<",\"wide_tiles\":"<<fallback<<",\"invalid_tiles\":"<<invalid
      <<",\"improved_tiles\":"<<improved<<",\"tied_tiles\":"<<ties<<",\"changed_tiles\":"<<changed
      <<",\"proxy_reduction_percent\":"<<(greedy_total?100*(1-exact_total/greedy_total):0)
      <<",\"rectangle_proxy_reduction_percent\":"<<(greedy_total?100*(1-rectangle_total/greedy_total):0)
      <<",\"rectangle_gap_tiles\":"<<rectangle_gap
      <<",\"host_oracle_and_setup_ms\":"<<oracle_setup
      <<",\"threadgroup_bytes\":"<<engine.solve.pipeline->staticThreadgroupMemoryLength()
      <<",\"thread_execution_width\":"<<engine.solve.pipeline->threadExecutionWidth()
      <<",\"gpu_max_threads\":"<<engine.solve.pipeline->maxTotalThreadsPerThreadgroup()<<",\"samples\":[";
    bool first=true;
    for(unsigned threads:{64u,128u,256u,512u}) {
      engine.run(batches,threads,0);for(auto& b:batches)b->validate();
      engine.run(batches,threads,5);for(auto& b:batches)b->validate();
      engine.run(batches,threads,6);for(auto& b:batches)b->validate(true);
      engine.run(batches,threads,7);for(auto& b:batches)b->validate(false,true);
      engine.run(batches,threads,8);for(auto& b:batches)b->validate(true,false,true);
    }
    if(!self) {
      for(unsigned warm=0;warm<warmups;++warm)
        for(unsigned threads:{64u,128u,256u,512u})for(unsigned mode:{0u,6u,7u,8u})engine.run(batches,threads,mode);
      // Rotate ordering of thread counts between samples; no validation in timing.
      const unsigned counts[]={64,128,256,512};
      for(unsigned sample=0;sample<samples;++sample)for(unsigned j=0;j<4;++j) {
        unsigned threads=counts[(j+sample)%4];
        for(unsigned mode:{0u,6u,7u,8u}) {
          auto time=engine.run(batches,threads,mode);
          out<<(first?"":",")<<"{\"mode\":\""<<(mode==8?"rectangle_greedy":mode==7?"greedy":mode==6?"rectangle":"complete")<<"\",\"threads\":"<<threads<<",\"sample\":"<<sample
            <<",\"gpu_ms\":"<<time.gpu_ms<<",\"wall_ms\":"<<time.wall_ms<<"}";first=false;
        }
      }
      engine.run(batches,128,5);
      const char* names[]={"prepare","dp_with_device_decisions","trace_from_device_decisions","wide_dispatch"};
      // Diagnostic split uses additional device decision traffic; it is not an
      // additive decomposition of the fused complete kernel above.
      for(unsigned sample=0;sample<samples;++sample)for(unsigned mode=1;mode<=4;++mode) {
        auto time=engine.run(batches,128,mode);
        out<<(first?"":",")<<"{\"mode\":\""<<names[mode-1]<<"\",\"threads\":128,\"sample\":"<<sample
          <<",\"gpu_ms\":"<<time.gpu_ms<<",\"wall_ms\":"<<time.wall_ms<<"}";first=false;
      }
      for(auto& b:batches)b->validate();
    }
    out<<"],\"validation\":\"passed: policy bits, integer conversion, frontier and rectangle exact objectives, CPU-greedy parity, hybrid maps and costs, buffer guards; combined and split kernels at 64/128/256/512 threads\"}\n";
    std::cerr<<"Validated "<<tiles<<" tiles, "<<fallback<<" wide, "<<invalid<<" invalid; "<<batches.size()<<" shapes\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"frontier: "<<e.what()<<"\n";return 1;}
}
