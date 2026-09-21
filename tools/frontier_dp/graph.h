// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "codec/ac_strategy_search_policy.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace frontier {
struct Node { uint32_t begin; uint16_t slot, count; };
struct Edge { uint16_t child, slot, candidate, pad; };
struct Rect { uint8_t x,y,w,h,strategy,pad[3]; };
struct Params {
  uint32_t tiles,candidates,nodes,slots,root,terminal,width,height;
  float multiplier;
  uint32_t discount, split, workers;
};
struct Meta { uint32_t shift, flags; }; // 1=wide fallback, 2=invalid input
struct Wide {
  std::array<uint64_t,5> limb{};
  bool operator==(const Wide&) const = default;
  bool operator<(const Wide& b) const {
    for(int i=4;i>=0;--i) if(limb[i]!=b.limb[i])return limb[i]<b.limb[i];
    return false;
  }
  Wide operator+(const Wide& b) const {
    Wide r; uint64_t carry=0;
    for(int i=0;i<5;++i) {
      const uint64_t a=limb[i]+b.limb[i], c=a+carry;
      carry=(a<limb[i])|(c<a); r.limb[i]=c;
    }
    if(carry)throw std::runtime_error("wide integer overflow");
    return r;
  }
};
static_assert(sizeof(Node)==8 && sizeof(Edge)==8 && sizeof(Rect)==8);
static_assert(sizeof(Params)==48 && sizeof(Wide)==40 && sizeof(Meta)==8);
struct Graph {
  unsigned w,h,slot_count=0,root=0,terminal=0;
  std::vector<Rect> rect;
  std::vector<Node> node;
  std::vector<Edge> edge;
  std::vector<uint16_t> level;
  std::array<uint32_t,66> offsets{};
  explicit Graph(unsigned width,unsigned height):w(width),h(height) {
    if(w<1||w>8||h<1||h>8)throw std::runtime_error("graph dimensions");
    std::array<std::vector<unsigned>,64> anchors;
    for(const auto& stage:gjxl::ac_strategy_internal::kCandidateStages) {
      auto d=gjxl::GetAcStrategyInfo(stage.strategy)->covered_blocks;
      for(unsigned y=0;y+d.height<=h;y+=stage.anchor_step)
        for(unsigned x=0;x+d.width<=w;x+=stage.anchor_step) {
          anchors[y*8+x].push_back(rect.size());
          rect.push_back({uint8_t(x),uint8_t(y),uint8_t(d.width),uint8_t(d.height),
                          uint8_t(stage.strategy),{0,0,0}});
        }
    }
    std::unordered_map<uint32_t,unsigned> memo;
    std::vector<std::vector<std::pair<unsigned,unsigned>>> outgoing;
    std::vector<unsigned> area;
    std::array<std::vector<unsigned>,65> levels;
    std::function<unsigned(std::array<uint8_t,8>)> visit=[&](auto s) {
      uint32_t key=0;unsigned a=0,y=h,x=0;
      for(unsigned j=0;j<w;++j) {key|=uint32_t(s[j])<<(4*j);a+=s[j];if(s[j]<y){y=s[j];x=j;}}
      if(auto it=memo.find(key);it!=memo.end())return it->second;
      const unsigned u=node.size();memo.emplace(key,u);node.push_back({});
      outgoing.emplace_back();area.push_back(a);levels[a].push_back(u);
      if(y==h){terminal=u;return u;}
      for(unsigned c:anchors[y*8+x]) {
        auto r=rect[c];bool fits=true;
        for(unsigned j=x;j<x+r.w;++j)fits&=s[j]==y;
        if(!fits)continue;
        auto child=s;for(unsigned j=x;j<x+r.w;++j)child[j]=y+r.h;
        unsigned v=visit(child);outgoing[u].push_back({c,v});
      }
      if(outgoing[u].empty())throw std::runtime_error("dead graph state");
      return u;
    };
    root=visit({});
    auto last=area;
    for(unsigned u=0;u<node.size();++u)
      for(auto [c,v]:outgoing[u])last[v]=std::min(last[v],area[u]);
    std::array<std::vector<unsigned>,65> release;
    for(unsigned u=0;u<node.size();++u)release[last[u]].push_back(u);
    std::vector<unsigned> free;
    for(int a=w*h;a>=0;--a) {
      for(unsigned u:levels[a]) {
        if(free.empty())node[u].slot=slot_count++;
        else {node[u].slot=free.back();free.pop_back();}
      }
      for(unsigned u:release[a])free.push_back(node[u].slot);
    }
    for(unsigned u=0;u<node.size();++u) {
      node[u].begin=edge.size();node[u].count=outgoing[u].size();
      for(auto[c,v]:outgoing[u])edge.push_back({uint16_t(v),node[v].slot,uint16_t(c),0});
    }
    for(unsigned a=0;a<=w*h;++a) {
      offsets[a]=level.size();for(auto u:levels[a])level.push_back(u);
    }
    offsets[w*h+1]=level.size();
    if(node.size()>12200||slot_count>1282||rect.size()>258)
      throw std::runtime_error("original graph capacity exceeded");
    if(w==8&&h==8&&(node.size()!=12200||edge.size()!=35511||slot_count!=1282||rect.size()!=258))
      throw std::runtime_error("source candidate policy changed");
  }
};
inline std::pair<uint32_t,uint32_t> Decompose(uint32_t bits) {
  const unsigned e=(bits>>23)&255;
  return {(bits&0x7fffff)|(e?0x800000:0),e?e-1:0};
}
inline Meta Normalize(const std::vector<float>& policy,std::vector<Wide>* wide) {
  unsigned shift=~0u,top=0,flags=0;
  for(float f:policy) {
    const uint32_t b=std::bit_cast<uint32_t>(f);
    if((b&0x7fffffff)>0x7f7fffff || ((b>>31)&& (b&0x7fffffff)))flags=2;
    auto[m,e]=Decompose(b);
    if(m){shift=std::min(shift,e+std::countr_zero(m));top=std::max(top,e+std::bit_width(m));}
  }
  if(shift==~0u)shift=0;
  if(!flags && top && top-shift>58)flags=1;
  wide->assign(policy.size(),{});
  if(flags==2)return {shift,flags};
  for(unsigned i=0;i<policy.size();++i) {
    auto[m,e]=Decompose(std::bit_cast<uint32_t>(policy[i]));
    if(!m)continue;
    unsigned trailing=std::countr_zero(m),s=e+trailing-shift;m>>=trailing;
    (*wide)[i].limb[s/64]=uint64_t(m)<<(s%64);
    if(s%64>40)(*wide)[i].limb[s/64+1]=uint64_t(m)>>(64-s%64);
  }
  return {shift,flags};
}
struct Solution { Wide cost; std::array<uint8_t,64> grid; };
template<class Cost> Solution OracleImpl(const Graph& g,const std::vector<Wide>& cost) {
  std::vector<Cost> value(g.node.size());std::vector<uint8_t> choice(g.node.size());
  for(int a=g.w*g.h-1;a>=0;--a)for(unsigned p=g.offsets[a];p<g.offsets[a+1];++p) {
    unsigned u=g.level[p];auto n=g.node[u];Cost best{};
    for(unsigned k=0;k<n.count;++k){auto e=g.edge[n.begin+k];Cost leaf;
      if constexpr(std::is_same_v<Cost,Wide>)leaf=cost[e.candidate];else leaf=cost[e.candidate].limb[0];
      Cost v=leaf+value[e.child];if(k==0||v<best){best=v;choice[u]=k;}}
    value[u]=best;
  }
  Solution out;out.grid.fill(255);
  if constexpr(std::is_same_v<Cost,Wide>)out.cost=value[g.root];else out.cost.limb[0]=value[g.root];
  for(unsigned u=g.root;u!=g.terminal;) {
    auto e=g.edge[g.node[u].begin+choice[u]];auto r=g.rect[e.candidate];
    for(unsigned y=0;y<r.h;++y)for(unsigned x=0;x<r.w;++x)
      out.grid[(r.y+y)*8+r.x+x]=(r.strategy<<1)|(!x&&!y);
    u=e.child;
  }
  return out;
}
inline Solution Oracle(const Graph& g,const std::vector<Wide>& cost,bool wide) {
  return wide?OracleImpl<Wide>(g,cost):OracleImpl<uint64_t>(g,cost);
}

// Independent subproblem family: rectangles produced by recursive straight
// cuts. Leaf placements must belong to the same candidate bank as frontier DP.
template<class Cost>
Solution RectangleOracleImpl(const Graph& g,const std::vector<Wide>& cost) {
  constexpr unsigned count=9*9*8*8;
  auto index=[](unsigned x,unsigned y,unsigned w,unsigned h) {
    return ((h*9+w)*8+y)*8+x;
  };
  std::vector<Cost> value(count);
  std::vector<uint16_t> choice(count,65535);
  for(unsigned c=0;c<g.rect.size();++c) {
    const auto r=g.rect[c];unsigned i=index(r.x,r.y,r.w,r.h);
    if constexpr(std::is_same_v<Cost,Wide>)value[i]=cost[c];
    else value[i]=cost[c].limb[0];
    choice[i]=c;
  }
  for(unsigned h=1;h<=g.h;++h)for(unsigned w=1;w<=g.w;++w)
    for(unsigned y=0;y+h<=g.h;++y)for(unsigned x=0;x+w<=g.w;++x) {
      unsigned i=index(x,y,w,h);
      const auto consider=[&](Cost candidate,uint16_t decision) {
        if(choice[i]==65535||candidate<value[i]){value[i]=candidate;choice[i]=decision;}
      };
      for(unsigned k=1;k<w;++k)
        consider(value[index(x,y,k,h)]+value[index(x+k,y,w-k,h)],512+k);
      for(unsigned k=1;k<h;++k)
        consider(value[index(x,y,w,k)]+value[index(x,y+k,w,h-k)],1024+k);
    }
  Solution out;out.grid.fill(255);
  if constexpr(std::is_same_v<Cost,Wide>)out.cost=value[index(0,0,g.w,g.h)];
  else out.cost.limb[0]=value[index(0,0,g.w,g.h)];
  std::function<void(unsigned,unsigned,unsigned,unsigned)> trace=[&](auto x,auto y,auto w,auto h) {
    unsigned decision=choice[index(x,y,w,h)];
    if(decision>=1024){unsigned k=decision-1024;trace(x,y,w,k);trace(x,y+k,w,h-k);}
    else if(decision>=512){unsigned k=decision-512;trace(x,y,k,h);trace(x+k,y,w-k,h);}
    else {
      auto r=g.rect[decision];
      for(unsigned dy=0;dy<h;++dy)for(unsigned dx=0;dx<w;++dx)
        out.grid[(y+dy)*8+x+dx]=(r.strategy<<1)|(!dx&&!dy);
    }
  };
  trace(0,0,g.w,g.h);return out;
}
inline Solution RectangleOracle(const Graph& g,const std::vector<Wide>& cost,bool wide) {
  return wide?RectangleOracleImpl<Wide>(g,cost):RectangleOracleImpl<uint64_t>(g,cost);
}

struct RectangleNode { uint32_t begin; uint16_t count,candidate; };
struct RectangleEdge { uint16_t left,right; };
struct RectangleGraph {
  std::vector<RectangleNode> node;
  std::vector<RectangleEdge> edge;
  std::array<uint32_t,18> offsets{};
  unsigned root=0;
  explicit RectangleGraph(const Graph& g) {
    auto index=[](unsigned x,unsigned y,unsigned w,unsigned h) {return ((h*9+w)*8+y)*8+x;};
    std::array<uint16_t,9*9*8*8> ids{},leaves;
    leaves.fill(65535);
    for(unsigned c=0;c<g.rect.size();++c){auto r=g.rect[c];leaves[index(r.x,r.y,r.w,r.h)]=c;}
    for(unsigned d=2;d<=g.w+g.h;++d) {
      offsets[d]=node.size();
      for(unsigned h=1;h<=g.h;++h)for(unsigned w=1;w<=g.w;++w)if(w+h==d)
        for(unsigned y=0;y+h<=g.h;++y)for(unsigned x=0;x+w<=g.w;++x) {
          const unsigned i=index(x,y,w,h);ids[i]=node.size();
          RectangleNode n{uint32_t(edge.size()),uint16_t(w+h-2),leaves[i]};
          node.push_back(n);
          for(unsigned k=1;k<w;++k)edge.push_back({ids[index(x,y,k,h)],ids[index(x+k,y,w-k,h)]});
          for(unsigned k=1;k<h;++k)edge.push_back({ids[index(x,y,w,k)],ids[index(x,y+k,w,h-k)]});
        }
    }
    offsets[g.w+g.h+1]=node.size();root=ids[index(0,0,g.w,g.h)];
    if(g.w==8&&g.h==8&&(node.size()!=1296||edge.size()!=6048))
      throw std::runtime_error("rectangle graph count mismatch");
  }
};
static_assert(sizeof(RectangleNode)==8&&sizeof(RectangleEdge)==4);
}
