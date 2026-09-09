// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once
#include <cstring>
#include <memory>
#include <stdexcept>
#include "codec/color_transform.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/ops/aq_evaluation.h"
#include "sparse_frame_fixture.h"

namespace resident_sparse_test {
using namespace gjxl;
using gjxl_test::Check;
inline void Require(bool b,const char* message) {if(!b)throw std::runtime_error(message);}
template<class T> bool EqualBits(std::span<const T>a,std::span<const T>b) {
  return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size_bytes())==0);
}
struct Output {
  std::vector<float> quant,block;
  std::vector<double> scores{99.25};
  double score=77.5;
  QuantizerParams quantizer{123,456};
  Image3FBuffer rgb;
  VarDctEncoderFrame frame;
  Output(Extent2D extent,size_t block_count)
      :quant(block_count,-55.25f),block(block_count,-55.25f),rgb(extent),frame(gjxl_test::MakeFrame(7,2,4)) {
    for(size_t c=0;c<3;++c)std::fill(rgb.plane(c).begin(),rgb.plane(c).end(),-55.25f);
  }
  std::vector<uint8_t> Bytes()const {std::vector<uint8_t>v;Check(EncodeVarDctCodestream(frame,{},&v));return v;}
};
inline void Equal(const Output&a,const Output&b) {
  Require(EqualBits<float>(a.quant,b.quant)&&EqualBits<float>(a.block,b.block)&&EqualBits<double>(a.scores,b.scores)&&
      std::memcmp(&a.score,&b.score,sizeof(double))==0&&a.quantizer.global_scale==b.quantizer.global_scale&&
      a.quantizer.quant_dc==b.quantizer.quant_dc,"Resident sparse host output differs");
  for(size_t c=0;c<3;++c)Require(EqualBits<float>(a.rgb.plane(c),b.rgb.plane(c)),"Resident sparse RGB differs bitwise");
  Require(a.Bytes()==b.Bytes(),"Resident sparse codestream differs");
  gjxl_test::EqualSparseCoefficients(a.frame,b.frame);
}
struct Fixture {
  FrameGeometry geometry;
  Extent2D extent,blocks,padded_extent;
  size_t block_count=0;
  AcStrategyGrid strategies;
  Image3FBuffer source,opsin;
  std::vector<uint8_t> sharpness;
  std::vector<float> field;
  float quant_dc=0;
  Fixture(Extent2D size={65,67},float amplitude=1.0f):extent(size),source(size) {
    Check(FrameGeometry::Create(size,&geometry));blocks=geometry.block_grid().blocks;padded_extent=geometry.padded_frame();
    block_count=blocks.width*blocks.height;sharpness.assign(block_count,4);field.assign(block_count,0.8f);
    Check(AcStrategyGrid::Create(blocks,&strategies));strategies.fill_dct8();
    uint32_t rng=0x7f4a7c15u;
    for(size_t c=0;c<3;++c)for(float&v:source.plane(c)) {
      rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;v=0.2f+amplitude*static_cast<float>(rng&65535)/65535.0f;
    }
    Image3FBuffer padded(padded_extent);opsin.resize(padded_extent);
    for(size_t c=0;c<3;++c)for(size_t y=0;y<padded_extent.height;++y)for(size_t x=0;x<padded_extent.width;++x)
      padded.view().plane[c].Row(y)[x]=source.view().plane[c].Row(std::min(y,extent.height-1))[std::min(x,extent.width-1)];
    Check(LinearRgbToOpsin(padded.const_view(),255.0f,opsin.view()));
    Check(ComputeInitialQuantDc(1.1f,&quant_dc));
  }
  std::unique_ptr<PreparedAqEvaluation> Prepare(GpuBackend&backend)const {
    std::unique_ptr<PreparedAqEvaluation>p;
    Check(PrepareAqEvaluation(backend,{.original_linear_rgb=source.const_view(),.coding_opsin=opsin.const_view(),
        .strategies=&strategies,.epf_sharpness={sharpness.data(),blocks,blocks.width},
        .resident_quantization=true,.coefficient_decision_mode=AcCoefficientDecisionMode::kAdjustedSharedQuant},&p));
    Check(p->PrepareInvariantColorCorrelationResident({field.data(),blocks,blocks.width},quant_dc));
    return p;
  }
  Output NewOutput()const{return Output(extent,block_count);}
  Status Run(PreparedAqEvaluation&p,Output&out,unsigned method)const {
    if(method==2) {
      AqEvaluationOutput::Final final{.reconstructed_linear_rgb=out.rgb.view(),.frame=&out.frame};
      return p.Evaluate({.quant_field={field.data(),blocks,blocks.width},.quant_dc=quant_dc},
          {.block_distance_map={out.block.data(),blocks,blocks.width},.score=&out.score,.quantizer=&out.quantizer,.final=&final});
    }
    const bool score=method==0;
    return p.EvaluateResidentButteraugliPolicy(
        {.adjusted_initial_quant_field={field.data(),blocks,blocks.width},.quant_dc=quant_dc,.butteraugli_target=1.1f,
         .lower_bound=0.1f,.upper_bound=10.0f,.iterations=1,.evaluate_final_field=score},
        {.quant_field={out.quant.data(),blocks,blocks.width},
         .block_distance_map=score?PlaneF32View{out.block.data(),blocks,blocks.width}:PlaneF32View{},
         .score_history=&out.scores,.reconstructed_linear_rgb=score?out.rgb.view():Image3FView{},.frame=&out.frame});
  }
};
}
