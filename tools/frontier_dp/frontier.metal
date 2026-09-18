// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;
struct Node { uint begin; ushort slot,count; };
struct Edge { ushort child,slot,candidate,pad; };
struct Rect { uchar x,y,w,h,strategy,pad[3]; };
struct Params {
  uint tiles,candidates,nodes,slots,root,terminal,width,height;
  float multiplier;uint discount,split,workers;
};
struct Meta { uint shift,flags; };
struct Wide { ulong limb[5]; };
uint Trailing(uint m) { return 31-clz(m & (0u-m)); }
uint Mantissa(uint b) {return (b&0x7fffffu)|(((b>>23)&255)?0x800000u:0u);}
uint Exponent(uint b) {uint e=(b>>23)&255;return e?e-1:0;}
uint PolicyBits(float raw,float multiplier) {
  uint bits=as_type<uint>(raw),a=bits&0x7fffffffu;
  if(a<0x1000000u) {
    // The DCT8 multiplier lies in [0.714...,1]. Below twice FLT_MIN,
    // form the correctly rounded product in units of 2^-149; hardware
    // multiplication would flush subnormal inputs or results on this GPU.
    uint b=as_type<uint>(multiplier),shift=149-Exponent(a)-Exponent(b);
    ulong product=ulong(Mantissa(a))*Mantissa(b),units=product>>shift;
    ulong tail=product&((1ul<<shift)-1),mid=1ul<<(shift-1);
    units+=tail>mid||(tail==mid&&(units&1));
    return (bits&0x80000000u)|uint(units);
  }
  return as_type<uint>(raw*multiplier);
}
Wide Convert(uint b,uint common) {
  Wide r={{0,0,0,0,0}};uint m=Mantissa(b);if(!m)return r;
  uint t=Trailing(m),s=Exponent(b)+t-common;m>>=t;
  r.limb[s/64]=ulong(m)<<(s%64);
  if(s%64>40)r.limb[s/64+1]=ulong(m)>>(64-s%64);
  return r;
}
Wide Add(Wide a,Wide b) {
  Wide r;ulong carry=0;
  for(uint i=0;i<5;++i){ulong s=a.limb[i]+b.limb[i],v=s+carry;carry=ulong(s<a.limb[i])|ulong(v<s);r.limb[i]=v;}
  return r;
}
bool Less(Wide a,Wide b) {for(int i=4;i>=0;--i)if(a.limb[i]!=b.limb[i])return a.limb[i]<b.limb[i];return false;}

// All subsequent arithmetic consumes the frozen binary32 policy bit patterns.
kernel void fdp_prepare(device const float* raw [[buffer(0)]],
    device const Rect* rect [[buffer(1)]],device uint* policy [[buffer(2)]],
    device ulong* normalized [[buffer(3)]],device Meta* meta [[buffer(4)]],
    constant Params& p [[buffer(5)]],uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint threads [[threads_per_threadgroup]]) {
  threadgroup uint shifts[256],tops[256],invalid[256];
  uint shift=~0u,top=0,bad=0;
  for(uint c=tid;c<p.candidates;c+=threads) {
    const uint i=tile*p.candidates+c;
    float f=raw[i];uint original=as_type<uint>(f);
    uint b=p.discount&&rect[c].strategy==0?PolicyBits(f,p.multiplier):original;policy[i]=b;
    if((original&0x7fffffffu)>0x7f7fffffu || ((original>>31) && (original&0x7fffffffu)))bad=1;
    uint m=Mantissa(b),e=Exponent(b);
    if(m){shift=min(shift,e+Trailing(m));top=max(top,e+32-clz(m));}
  }
  shifts[tid]=shift;tops[tid]=top;invalid[tid]=bad;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint stride=threads/2;stride;stride/=2) {
    if(tid<stride){shifts[tid]=min(shifts[tid],shifts[tid+stride]);tops[tid]=max(tops[tid],tops[tid+stride]);invalid[tid]|=invalid[tid+stride];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint common=shifts[0]==~0u?0:shifts[0];
  uint flags=invalid[0]?2u:((tops[0] && tops[0]-common>58)?1u:0u);
  if(tid==0)meta[tile]={common,flags};
  // Device visibility for policy values written by other lanes is not needed:
  // each lane converts exactly the candidates it wrote above.
  for(uint c=tid;c<p.candidates;c+=threads) {
    uint i=tile*p.candidates+c;
    normalized[i]=flags?0:Convert(policy[i],common).limb[0];
  }
}

kernel void fdp_solve(device const Node* nodes [[buffer(0)]],
    device const Edge* edges [[buffer(1)]],device const ushort* levels [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],device const Rect* rect [[buffer(4)]],
    device const ulong* costs [[buffer(5)]],device const Meta* meta [[buffer(6)]],
    device Wide* result [[buffer(7)]],device uchar* grid [[buffer(8)]],
    device uchar* saved_choice [[buffer(9)]],constant Params& p [[buffer(10)]],
    uint tile [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if(meta[tile].flags)return; // Uniform for the entire group, before barriers.
  threadgroup ulong value[1282],candidate[258];
  threadgroup uchar choice[12200];
  for(uint c=tid;c<p.candidates;c+=threads)candidate[c]=costs[tile*p.candidates+c];
  if(tid==0)value[nodes[p.terminal].slot]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int a=int(p.width*p.height)-1;a>=0;--a) {
    for(uint i=offsets[a]+tid;i<offsets[a+1];i+=threads) {
      uint u=levels[i];Node n=nodes[u];ulong best=0;uchar pick=0;
      for(ushort k=0;k<n.count;++k) {
        Edge e=edges[n.begin+k];ulong trial=candidate[e.candidate]+value[e.slot];
        if(k==0||trial<best){best=trial;pick=uchar(k);}
      }
      value[n.slot]=best;choice[u]=pick;
      if(p.split)saved_choice[tile*p.nodes+u]=pick;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(tid==0) {
    result[tile]={{value[nodes[p.root].slot],0,0,0,0}};
    if(!p.split) {
      for(uint i=0;i<64;++i)grid[tile*64+i]=255;
      for(uint u=p.root;u!=p.terminal;) {
        Edge e=edges[nodes[u].begin+choice[u]];Rect r=rect[e.candidate];
        for(uint y=0;y<r.h;++y)for(uint x=0;x<r.w;++x)
          grid[tile*64+(r.y+y)*8+r.x+x]=(r.strategy<<1)|uint(x==0&&y==0);
        u=e.child;
      }
    }
  }
}

kernel void fdp_trace(device const Node* nodes [[buffer(0)]],
    device const Edge* edges [[buffer(1)]],device const Rect* rect [[buffer(2)]],
    device const Meta* meta [[buffer(3)]],device const uchar* choice [[buffer(4)]],
    device uchar* grid [[buffer(5)]],constant Params& p [[buffer(6)]],
    uint tile [[thread_position_in_grid]]) {
  if(tile>=p.tiles||meta[tile].flags)return;
  for(uint i=0;i<64;++i)grid[tile*64+i]=255;
  for(uint u=p.root;u!=p.terminal;) {
    Edge e=edges[nodes[u].begin+choice[tile*p.nodes+u]];Rect r=rect[e.candidate];
    for(uint y=0;y<r.h;++y)for(uint x=0;x<r.w;++x)
      grid[tile*64+(r.y+y)*8+r.x+x]=(r.strategy<<1)|uint(x==0&&y==0);
    u=e.child;
  }
}

// Bounded persistent workers use device memory. No CPU readback is needed to
// discover exceptional tiles. Each worker processes its own disjoint tile set.
kernel void fdp_wide(device const Node* nodes [[buffer(0)]],
    device const Edge* edges [[buffer(1)]],device const ushort* levels [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],device const Rect* rect [[buffer(4)]],
    device const uint* policy [[buffer(5)]],device const Meta* meta [[buffer(6)]],
    device Wide* result [[buffer(7)]],device uchar* grid [[buffer(8)]],
    device Wide* scratch [[buffer(9)]],device uchar* decisions [[buffer(10)]],
    constant Params& p [[buffer(11)]],uint worker [[thread_position_in_grid]]) {
  if(worker>=p.workers)return;
  device Wide* value=scratch+worker*p.slots;
  device uchar* choice=decisions+worker*p.nodes;
  for(uint tile=worker;tile<p.tiles;tile+=p.workers) {
    if(meta[tile].flags!=1)continue;
    uint common=meta[tile].shift;value[nodes[p.terminal].slot]={{0,0,0,0,0}};
    for(int a=int(p.width*p.height)-1;a>=0;--a)
      for(uint i=offsets[a];i<offsets[a+1];++i) {
        uint u=levels[i];Node n=nodes[u];Wide best={{0,0,0,0,0}};uchar pick=0;
        for(ushort k=0;k<n.count;++k) {
          Edge e=edges[n.begin+k];
          Wide trial=Add(Convert(policy[tile*p.candidates+e.candidate],common),value[e.slot]);
          if(k==0||Less(trial,best)){best=trial;pick=uchar(k);}
        }
        value[n.slot]=best;choice[u]=pick;
      }
    result[tile]=value[nodes[p.root].slot];
    for(uint i=0;i<64;++i)grid[tile*64+i]=255;
    for(uint u=p.root;u!=p.terminal;) {
      Edge e=edges[nodes[u].begin+choice[u]];Rect r=rect[e.candidate];
      for(uint y=0;y<r.h;++y)for(uint x=0;x<r.w;++x)
        grid[tile*64+(r.y+y)*8+r.x+x]=(r.strategy<<1)|uint(x==0&&y==0);
      u=e.child;
    }
  }
}

struct RectangleNode { uint begin;ushort count,candidate; };
struct RectangleEdge { ushort left,right; };
kernel void rectangle_solve(device const RectangleNode* nodes [[buffer(0)]],
    device const RectangleEdge* edges [[buffer(1)]],device const uint* offsets [[buffer(2)]],
    device const Rect* rect [[buffer(3)]],device const ulong* costs [[buffer(4)]],
    device const Meta* meta [[buffer(5)]],device Wide* result [[buffer(6)]],
    device uchar* grid [[buffer(7)]],constant Params& p [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if(meta[tile].flags)return;
  threadgroup ulong value[1296],candidate[258];
  threadgroup ushort choice[1296];
  for(uint c=tid;c<p.candidates;c+=threads)candidate[c]=costs[tile*p.candidates+c];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint d=2;d<=p.width+p.height;++d) {
    for(uint u=offsets[d]+tid;u<offsets[d+1];u+=threads) {
      RectangleNode n=nodes[u];ushort pick=n.candidate;
      ulong best=pick==65535?0:candidate[pick];
      for(ushort k=0;k<n.count;++k) {
        RectangleEdge e=edges[n.begin+k];ulong trial=value[e.left]+value[e.right];
        if(pick==65535||trial<best){best=trial;pick=0x8000u|k;}
      }
      value[u]=best;choice[u]=pick;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(tid==0) {
    result[tile]={{value[p.root],0,0,0,0}};
    for(uint i=0;i<64;++i)grid[tile*64+i]=255;
    ushort stack[64];uint size=1;stack[0]=p.root;
    while(size) {
      ushort u=stack[--size],pick=choice[u];
      if(pick&0x8000u){RectangleEdge e=edges[nodes[u].begin+(pick&0x7fffu)];stack[size++]=e.right;stack[size++]=e.left;}
      else {Rect r=rect[pick];
        for(uint y=0;y<r.h;++y)for(uint x=0;x<r.w;++x)
          grid[tile*64+(r.y+y)*8+r.x+x]=(r.strategy<<1)|uint(x==0&&y==0);}
    }
  }
}

kernel void rectangle_wide(device const RectangleNode* nodes [[buffer(0)]],
    device const RectangleEdge* edges [[buffer(1)]],device const Rect* rect [[buffer(2)]],
    device const uint* policy [[buffer(3)]],device const Meta* meta [[buffer(4)]],
    device Wide* result [[buffer(5)]],device uchar* grid [[buffer(6)]],
    device Wide* scratch [[buffer(7)]],device ushort* decisions [[buffer(8)]],
    constant Params& p [[buffer(9)]],uint worker [[thread_position_in_grid]]) {
  if(worker>=p.workers)return;
  device Wide* value=scratch+worker*p.nodes;
  device ushort* choice=decisions+worker*p.nodes;
  for(uint tile=worker;tile<p.tiles;tile+=p.workers) {
    if(meta[tile].flags!=1)continue;
    for(uint u=0;u<p.nodes;++u) {
      RectangleNode n=nodes[u];ushort pick=n.candidate;
      Wide best={{0,0,0,0,0}};
      if(pick!=65535)best=Convert(policy[tile*p.candidates+pick],meta[tile].shift);
      for(ushort k=0;k<n.count;++k) {
        RectangleEdge e=edges[n.begin+k];Wide trial=Add(value[e.left],value[e.right]);
        if(pick==65535||Less(trial,best)){best=trial;pick=0x8000u|k;}
      }
      value[u]=best;choice[u]=pick;
    }
    result[tile]=value[p.root];
    for(uint i=0;i<64;++i)grid[tile*64+i]=255;
    ushort stack[64];uint size=1;stack[0]=p.root;
    while(size) {
      ushort u=stack[--size],pick=choice[u];
      if(pick&0x8000u){RectangleEdge e=edges[nodes[u].begin+(pick&0x7fffu)];stack[size++]=e.right;stack[size++]=e.left;}
      else {Rect r=rect[pick];
        for(uint y=0;y<r.h;++y)for(uint x=0;x<r.w;++x)
          grid[tile*64+(r.y+y)*8+r.x+x]=(r.strategy<<1)|uint(x==0&&y==0);}
    }
  }
}

// Literal ordinary-bank CPU traversal. One lane owns a tile; all floating
// additions retain the CPU order. This is a policy-preserving control.
constant uchar greedy_width[7]={1,1,2,2,2,4,4};
constant uchar greedy_height[7]={1,2,1,2,4,2,4};
constant uchar greedy_strategy[7]={0,6,7,4,10,11,5};
struct GreedyTile {
  uint cost[64];uchar priority[64],grid[64];uint w,h;
  device const uint* policy;device const ushort* lookup;
};
uint GreedyCost(thread GreedyTile& g,uint s,uint x,uint y) {
  return g.policy[g.lookup[s*64+y*8+x]]&0x7fffffffu;
}
uint GreedyAdd(uint a,uint b) {
  if(!a)return b;if(!b)return a;
  if(a>=0x800000u&&b>=0x800000u)return as_type<uint>(as_type<float>(a)+as_type<float>(b));
  // Metal FP32 arithmetic flushes subnormals. Preserve CPU round-to-nearest,
  // ties-to-even with three guard/sticky bits only on this exceptional path.
  if(a<b){uint t=a;a=b;b=t;}if(a>=0x7f800000u)return a;
  uint e=Exponent(a),delta=e-Exponent(b);
  ulong small=ulong(Mantissa(b))<<3;
  small=delta>=64?ulong(small!=0):(delta?((small>>delta)|ulong((small&((1ul<<delta)-1))!=0)):small);
  ulong sum=(ulong(Mantissa(a))<<3)+small;
  if(sum>=(1ul<<27)){sum=(sum>>1)|(sum&1);++e;}
  uint m=uint(sum>>3),tail=uint(sum&7);m+=tail>4||(tail==4&&(m&1));
  if(m>=0x1000000u){m>>=1;++e;}
  if(e>=254)return 0x7f800000u;
  return m<0x800000u?m:((e+1)<<23)|(m&0x7fffffu);
}
// Keep the mutating helpers out of line. The tested Metal compiler produced
// incorrect maps with inlining on; shader validation masked that failure.
// Both ordinary and instrumented kernels are checked against the CPU policy.
__attribute__((noinline)) void GreedySet(thread GreedyTile& g,uint s,uint x,uint y,uint cost) {
  for(uint dy=0;dy<greedy_height[s];++dy)for(uint dx=0;dx<greedy_width[s];++dx) {
    uint i=(y+dy)*8+x+dx;g.grid[i]=(s<<1)|uint(dx==0&&dy==0);g.cost[i]=0;
  }
  g.cost[y*8+x]=cost;
}
bool GreedyCrossH(thread GreedyTile& g,uint start,uint y,uint end) {
  if(start>=g.w||y>=g.h||y==0)return false;
  while(start&&!(g.grid[y*8+start]&1))--start;
  for(uint x=start;x<min(end,g.w);) {
    uchar c=g.grid[y*8+x];if(!(c&1))return true;x+=greedy_width[c>>1];
  }
  return false;
}
bool GreedyCrossV(thread GreedyTile& g,uint x,uint start,uint end) {
  if(x>=g.w||start>=g.h||x==0)return false;
  while(start&&!(g.grid[start*8+x]&1))--start;
  for(uint y=start;y<min(end,g.h);) {
    uchar c=g.grid[y*8+x];if(!(c&1))return true;y+=greedy_height[c>>1];
  }
  return false;
}
__attribute__((noinline)) void GreedyDivision(thread GreedyTile& g,uint blocks,uint x,uint y) {
  if(GreedyCrossH(g,x,y,x+blocks)||GreedyCrossH(g,x,y+blocks,x+blocks)||
     GreedyCrossV(g,x,y,y+blocks)||GreedyCrossV(g,x+blocks,y,y+blocks))return;
  uint split_size=blocks/2,vertical=blocks==2?1:4,horizontal=blocks==2?2:5,square=blocks==2?3:6;
  bool av=!GreedyCrossV(g,x+split_size,y,y+blocks),ah=!GreedyCrossH(g,x,y+split_size,x+blocks);
  uint q[4]={0,0,0,0};
  for(uint dy=0;dy<blocks;++dy)for(uint dx=0;dx<blocks;++dx) {
    uint i=(dy/split_size)*2+dx/split_size;q[i]=GreedyAdd(q[i],g.cost[(y+dy)*8+x+dx]);}
  uint vl=0x7f7fffffu,vr=0x7f7fffffu,ht=0x7f7fffffu,hb=0x7f7fffffu;
  if(av) {
    if((g.grid[y*8+x]>>1)!=vertical)vl=GreedyCost(g,vertical,x,y);
    if((g.grid[y*8+x+split_size]>>1)!=vertical)vr=GreedyCost(g,vertical,x+split_size,y);
  }
  if(ah) {
    if((g.grid[y*8+x]>>1)!=horizontal)ht=GreedyCost(g,horizontal,x,y);
    if((g.grid[(y+split_size)*8+x]>>1)!=horizontal)hb=GreedyCost(g,horizontal,x,y+split_size);
  }
  uint sq=GreedyCost(g,square,x,y);
  uint q02=GreedyAdd(q[0],q[2]),q13=GreedyAdd(q[1],q[3]);
  uint q01=GreedyAdd(q[0],q[1]),q23=GreedyAdd(q[2],q[3]);
  uint vc=GreedyAdd(min(vl,q02),min(vr,q13));
  uint hc=GreedyAdd(min(ht,q01),min(hb,q23));
  if(sq<vc&&sq<hc)GreedySet(g,square,x,y,sq);
  else if(vc<hc) {
    if(vl<q02)GreedySet(g,vertical,x,y,vl);
    if(vr<q13)GreedySet(g,vertical,x+split_size,y,vr);
  } else {
    if(ht<q01)GreedySet(g,horizontal,x,y,ht);
    if(hb<q23)GreedySet(g,horizontal,x,y+split_size,hb);
  }
}
void GreedyMerge(thread GreedyTile& g,uint s,uint x,uint y,uint priority) {
  uint current=0;
  for(uint dy=0;dy<greedy_height[s];++dy)for(uint dx=0;dx<greedy_width[s];++dx) {
    uint i=(y+dy)*8+x+dx;if(g.priority[i]>=priority)return;current=GreedyAdd(current,g.cost[i]);
  }
  uint candidate=GreedyCost(g,s,x,y);if(candidate>=current)return;
  GreedySet(g,s,x,y,candidate);
  for(uint dy=0;dy<greedy_height[s];++dy)for(uint dx=0;dx<greedy_width[s];++dx)
    g.priority[(y+dy)*8+x+dx]=priority;
}
kernel void greedy_solve(device const uint* policy [[buffer(0)]],
    device const ushort* lookup [[buffer(1)]],device const Meta* meta [[buffer(2)]],
    device uchar* grid [[buffer(3)]],constant Params& p [[buffer(4)]],
    device Wide* result [[buffer(5)]],
    uint tile [[thread_position_in_grid]]) {
  if(tile>=p.tiles||meta[tile].flags==2)return;
  GreedyTile g;g.w=p.width;g.h=p.height;g.policy=policy+tile*p.candidates;g.lookup=lookup;
  for(uint i=0;i<64;++i){g.cost[i]=0;g.priority[i]=0;g.grid[i]=255;}
  for(uint y=0;y<g.h;++y)for(uint x=0;x<g.w;++x){g.cost[y*8+x]=GreedyCost(g,0,x,y);g.grid[y*8+x]=1;}
  const uint stages[4]={1,2,5,4};
  for(uint stage=0;stage<4;++stage) {
    uint s=stages[stage];
    for(uint y=0;y+greedy_height[s]<=g.h;y+=greedy_height[s])
      for(uint x=0;x+greedy_width[s]<=g.w;x+=greedy_width[s]) {
        if(y+3<g.h&&x+3<g.w) {
          if(s==5){if(((y|x)%4)==0)GreedyDivision(g,4,x,y);continue;}
          if(s==4)continue;
        }
        if((s==5&&y%4!=0)||(s==4&&x%4!=0))continue;
        if(y+1<g.h&&x+1<g.w) {
          if(s==2){if(((y|x)%2)==0)GreedyDivision(g,2,x,y);continue;}
          if(s==1)continue;
        }
        if((s==2&&y%2==1)||(s==1&&x%2==1))continue;
        GreedyMerge(g,s,x,y,stage<2?2:4);
      }
  }
  for(uint y=0;y+1<g.h;++y)for(uint x=0;x+1<g.w;++x)if((y|x)%2!=0)GreedyDivision(g,2,x,y);
  for(uint y=0;y+3<g.h;y+=2)for(uint x=0;x+3<g.w;x+=2)if((y|x)%4!=0)GreedyDivision(g,4,x,y);
  if(p.split) {
    Wide total={{0,0,0,0,0}};
    for(uint y=0;y<g.h;++y)for(uint x=0;x<g.w;++x)if(g.grid[y*8+x]&1)
      total=Add(total,Convert(GreedyCost(g,g.grid[y*8+x]>>1,x,y),meta[tile].shift));
    if(Less(result[tile],total))return;
    result[tile]=total;
  }
  for(uint i=0;i<64;++i)grid[tile*64+i]=g.grid[i]==255?255:
    (greedy_strategy[g.grid[i]>>1]<<1)|(g.grid[i]&1);
}
