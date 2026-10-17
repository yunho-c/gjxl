// SPDX-License-Identifier: Apache-2.0
// Private, native-endian research archive. Never a codestream or public API.
#pragma once
#include "gpu/ops/modular_resident_histogram_fitting.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace gjxl::frontier_diagnostic {
using namespace histogram_gpu_internal;
using namespace modular_internal;
struct Archive {
  std::vector<HistogramTrainingSamples> samples;
  std::vector<HistogramTreeLearningOptions> options;
  std::vector<uint32_t> ids;
};
template<class T> void Write(std::ostream &out, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}
template<class T> T Read(std::istream &in) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{}; in.read(reinterpret_cast<char *>(&value), sizeof(T)); return value;
}
template<class T> void WriteVector(std::ostream &out, const T &v) {
  Write(out, uint64_t(v.size()));
  if (!v.empty()) out.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(typename T::value_type));
}
template<class T> void ReadVector(std::istream &in, T &v, size_t limit) {
  const auto n = Read<uint64_t>(in);
  if (n > limit) throw std::runtime_error("Frontier archive vector exceeds bound");
  v.resize(n);
  if (n) in.read(reinterpret_cast<char *>(v.data()), n * sizeof(typename T::value_type));
}
inline void Save(const ResidentHistogramInput &input, const std::filesystem::path &path) {
  if (std::filesystem::exists(path)) throw std::runtime_error("Frontier archive already exists");
  std::ofstream out(path, std::ios::binary); out.exceptions(std::ios::badbit | std::ios::failbit);
  Write(out, uint64_t{0x31454c504d534647}); // GFSM PLE1, private schema 1.
  Write(out, uint64_t(input.requests.size()));
  for (const auto &r : input.requests) {
    const auto &s = *r.source.samples; const auto &o = r.learning;
    if (!o.multipliers.empty() || o.execution_stats) throw std::runtime_error("Unsupported archive policy");
    Write(out, r.source.source_id);
    Write(out, uint64_t(s.total_samples)); Write(out, uint64_t(s.predictor_count));
    Write(out, uint64_t(s.property_count)); Write(out, uint64_t(s.static_property_count));
    Write(out, s.predictors); Write(out, s.properties);
    Write(out, o.split_threshold); Write(out, o.fast_decode_multiplier); Write(out, o.static_range);
    Write(out, uint64_t(o.maximum_samples)); Write(out, uint64_t(o.maximum_leaves));
    Write(out, uint64_t(o.maximum_depth)); Write(out, uint64_t(o.cpu_participants));
    WriteVector(out, s.weights);
    for (const auto &v : s.residuals) WriteVector(out, v);
    for (const auto &v : s.static_values) WriteVector(out, v);
    for (const auto &v : s.values) WriteVector(out, v);
    for (const auto &v : s.thresholds) WriteVector(out, v);
  }
}
inline Archive Load(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary); in.exceptions(std::ios::badbit | std::ios::failbit);
  if (Read<uint64_t>(in) != 0x31454c504d534647) throw std::runtime_error("Invalid frontier archive magic");
  const auto n = Read<uint64_t>(in);
  if (!n || n > kMaximumResidentHistogramFits) throw std::runtime_error("Invalid frontier archive fit count");
  Archive a; a.samples.resize(n); a.options.resize(n); a.ids.resize(n);
  for (size_t i = 0; i < n; ++i) {
    auto &s = a.samples[i]; auto &o = a.options[i];
    a.ids[i] = Read<uint32_t>(in);
    s.total_samples = Read<uint64_t>(in); s.predictor_count = Read<uint64_t>(in);
    s.property_count = Read<uint64_t>(in); s.static_property_count = Read<uint64_t>(in);
    s.predictors = Read<decltype(s.predictors)>(in); s.properties = Read<decltype(s.properties)>(in);
    o.split_threshold = Read<float>(in); o.fast_decode_multiplier = Read<float>(in);
    o.static_range = Read<HistogramStaticRange>(in);
    o.maximum_samples = Read<uint64_t>(in); o.maximum_leaves = Read<uint64_t>(in);
    o.maximum_depth = Read<uint64_t>(in); o.cpu_participants = Read<uint64_t>(in);
    ReadVector(in, s.weights, kMaximumResidentHistogramRowsPerFit);
    for (auto &v : s.residuals) ReadVector(in, v, kMaximumResidentHistogramRowsPerFit);
    for (auto &v : s.static_values) ReadVector(in, v, kMaximumResidentHistogramRowsPerFit);
    for (auto &v : s.values) ReadVector(in, v, kMaximumResidentHistogramRowsPerFit);
    for (auto &v : s.thresholds) ReadVector(in, v, 256);
    if (auto status = ValidateHistogramTrainingSamples(s, o); !status.ok())
      throw std::runtime_error(std::string(status.message()));
  }
  return a;
}
} // namespace gjxl::frontier_diagnostic
