// Included after the selected native streams have been emitted and assembled.
// Re-emission must match their bit counts AND bytes before attribution is used.
if (const char* rate_path = std::getenv("GJXL_FINAL_FIELD_RATE_DUMP")) {
  std::array<uint64_t, 4> component{};
  double q_bits = 0.0;
  size_t q_tokens = 0;
  std::map<int32_t, size_t> residuals;
  for (size_t g = 0; g < dc_groups.size(); ++g) {
    BitWriter checked;
    size_t before = 0;
    auto require = [](Status s) { if (!s.ok()) std::abort(); };
    require(WriteSimpleDcGroupModularHeader(&checked, frame.profile().extra_dc_precision));
    component[0] += checked.bits_written() - before; before = checked.bits_written();
    require(WriteValidatedTokenStream(dc_streams[2*g], selected_dc_code, &checked));
    component[1] += checked.bits_written() - before; before = checked.bits_written();
    require(WriteSimpleAcMetadataModularHeader(dc_groups[g].block_extent,
        dc_groups[g].transform_anchor_count, &checked));
    component[2] += checked.bits_written() - before; before = checked.bits_written();
    require(WriteValidatedTokenStream(dc_streams[2*g+1], selected_dc_code, &checked));
    component[3] += checked.bits_written() - before;
    if (checked.bits_written() != common_sections[g+1].bits_written() ||
        !std::ranges::equal(checked.padded_bytes(), common_sections[g+1].padded_bytes()))
      std::abort();
    // Metadata leaves remain distinct when the DC tree is pruned.
    const auto& metadata = dc_streams[2*g+1];
    size_t group_q_tokens = 0;
    for (size_t t = 0; t < metadata.size(); ++t) {
      const auto token = metadata[t];
      bool is_quantizer = false;
      for (uint32_t c = 3; c <= 6; ++c)
        is_quantizer |= token.context == (dc_layout.full_tree() ? c : dc_layout.context_map[c]);
      if (!is_quantizer) continue;
      ++group_q_tokens;
      const size_t cluster = selected_dc_code.context_map[token.context];
      HybridUintToken hybrid;
      require(EncodeHybridUint(token.value, selected_dc_code.uint_configs[cluster], &hybrid));
      q_bits += hybrid.extra_bit_count;
      if (selected_dc_code.mode == EntropyCodingMode::kAns) {
        const auto frequency = selected_dc_code.ans_histograms[cluster].frequencies[hybrid.symbol];
        if (frequency == 0) std::abort();
        q_bits += std::log2(4096.0 / frequency);
      } else {
        q_bits += selected_dc_code.prefix_codes[cluster].depths[hybrid.symbol];
      }
      ++q_tokens;
      const uint32_t v = metadata[t].value;
      ++residuals[(v & 1u) ? -int32_t((v + 1u) / 2u) : int32_t(v / 2u)];
    }
    if (group_q_tokens != dc_groups[g].transform_anchor_count) std::abort();
  }
  uint64_t ac_bits = 0;
  for (const auto& section : ac_sections) ac_bits += section.bits_written();
  const uint64_t accounted = common_sections[0].bits_written() + ac_bits +
    component[0] + component[1] + component[2] + component[3];
  if (accounted > candidate_output.size() * 8u) std::abort();
  std::ofstream rate(rate_path); rate << std::setprecision(17);
  rate << "{\"total_bytes\":" << candidate_output.size()
       << ",\"dc_global_bits\":" << common_sections[0].bits_written()
       << ",\"dc_header_bits\":" << component[0]
       << ",\"dc_token_bits\":" << component[1]
       << ",\"metadata_header_bits\":" << component[2]
       << ",\"metadata_token_bits\":" << component[3]
       << ",\"ac_section_bits\":" << ac_bits
       << ",\"frame_toc_padding_bits\":" << candidate_output.size()*8u-accounted
       << ",\"dc_shared_model_bits\":" << selected_dc_cost.model_bits
       << ",\"ac_model_bits\":" << selected_ac_cost.model_bits
       << ",\"order_model_bits\":" << (selected.custom_order ? selected_order_cost.model_bits : 0)
       << ",\"quantizer_token_estimated_bits\":" << q_bits
       << ",\"quantizer_token_count\":" << q_tokens
       << ",\"dc_ans\":" << (selected_dc_code.mode == EntropyCodingMode::kAns ? "true" : "false")
       << ",\"reemit_byte_identity\":true,\"quantizer_residual_histogram\":{";
  bool first = true;
  for (auto [residual, count] : residuals) {
    if (!first) rate << ','; first = false;
    rate << '"' << residual << "\":" << count;
  }
  rate << "},\"global_scale\":" << frame.quantizer().params().global_scale
       << ",\"quant_dc\":" << frame.quantizer().params().quant_dc
       << ",\"extra_dc_precision\":" << unsigned(frame.profile().extra_dc_precision)
       << ",\"dc_smoothing\":" << (frame.profile().adaptive_dc_smoothing ? "true" : "false");
  auto add = [](uint64_t& h, uint64_t value) {
    for (int i = 0; i < 8; ++i) { h ^= (value >> (8*i)) & 255; h *= 1099511628211ULL; }
  };
  auto plane_hash = [&](const auto& plane) {
    uint64_t h = 14695981039346656037ULL;
    add(h, plane.extent.width); add(h, plane.extent.height);
    for (size_t y = 0; y < plane.extent.height; ++y)
      for (size_t x = 0; x < plane.extent.width; ++x)
        add(h, static_cast<uint64_t>(plane.Row(y)[x]));
    return h;
  };
  rate << ",\"raw_quant_fingerprint\":\"" << plane_hash(frame.raw_quant_field())
       << "\",\"sharpness_fingerprint\":\"" << plane_hash(frame.epf_sharpness())
       << "\",\"ytox_fingerprint\":\"" << plane_hash(frame.color_correlation().y_to_x_map())
       << "\",\"ytob_fingerprint\":\"" << plane_hash(frame.color_correlation().y_to_b_map()) << "\"";
  const auto qdc = frame.quantized_dc();
  rate << ",\"quantized_dc_fingerprints\":[";
  for (size_t c=0;c<3;++c) { if(c)rate<<',';rate << '"' << plane_hash(qdc.plane[c]) << '"'; }
  std::array<uint64_t,3> ac_hash{14695981039346656037ULL,14695981039346656037ULL,14695981039346656037ULL};
  std::array<uint64_t,3> ac_nonzero{};
  uint64_t ac_count=0;
  for (size_t g=0;g<frame.ac_group_count();++g) {
    VarDctAcGroupView group;
    if (!frame.GetAcGroup(g,&group).ok()) std::abort();
    ac_count += group.used_coefficient_count;
    for (size_t c=0;c<3;++c) {
      add(ac_hash[c],group.used_coefficient_count);
      for (size_t i=0;i<group.used_coefficient_count;++i) {
        const int32_t value=group.coefficients[c][i];
        add(ac_hash[c],static_cast<uint64_t>(value));
        ac_nonzero[c] += value!=0;
      }
    }
  }
  rate << "],\"ac_coefficient_fingerprints\":[";
  for(size_t c=0;c<3;++c) {if(c)rate<<',';rate<<'"'<<ac_hash[c]<<'"';}
  rate << "],\"ac_nonzero_counts\":[";
  for(size_t c=0;c<3;++c) {if(c)rate<<',';rate<<ac_nonzero[c];}
  rate << "],\"ac_coefficient_count_per_channel\":" << ac_count;
  std::map<int32_t,size_t> quant_histogram;
  for(size_t y=0;y<frame.raw_quant_field().extent.height;++y)
    for(size_t x=0;x<frame.raw_quant_field().extent.width;++x)
      ++quant_histogram[frame.raw_quant_field().Row(y)[x]];
  rate << ",\"raw_quant_histogram\":{";
  bool first_quant=true;
  for(auto [q,n]:quant_histogram) {if(!first_quant)rate<<',';first_quant=false;rate<<'"'<<q<<"\":"<<n;}
  rate << "}}\n";
}
