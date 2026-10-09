// SPDX-License-Identifier: Apache-2.0
// Exact token experiment, adapted from the qualified Metal integer algorithm.
#include <cuda_runtime.h>

#include <climits>

#include "gpu/cuda/cuda_ac_tokenization_kernels.h"
namespace gjxl::cuda_internal {
namespace {
using uint = uint32_t;
using ushort = uint16_t;
using uchar = uint8_t;
__device__ __forceinline__ uint WarpSum(uint value) {
  for (int delta = 16; delta; delta /= 2)
    value += __shfl_down_sync(0xffffffffu, value, delta);
  return __shfl_sync(0xffffffffu, value, 0);
}
__device__ __forceinline__ uint WarpMax(uint value) {
  for (int delta = 16; delta; delta /= 2)
    value = max(value, __shfl_down_sync(0xffffffffu, value, delta));
  return __shfl_sync(0xffffffffu, value, 0);
}
__device__ __forceinline__ uint WarpPrefix(uint value) {
  const uint original = value, lane = threadIdx.x & 31;
  for (int delta = 1; delta < 32; delta *= 2) {
    const uint other = __shfl_up_sync(0xffffffffu, value, delta);
    if (lane >= delta) value += other;
  }
  return value - original;
}
__device__ __constant__ uint16_t kCoefficientFrequencyContext[64] = {
    0xBAD, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15,    15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
    23,    23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
    27,    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
};
__device__ __constant__ uint16_t kCoefficientNonzeroContext[64] = {
    0xBAD, 0,   31,  62,  62,  93,  93,  93,  93,  123, 123, 123, 123,
    152,   152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
    180,   180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
};
__device__ __forceinline__ uint ac_token_channel(uint task) {
  const uint c = task % 3;
  return c < 2 ? c ^ 1u : 2u;
}
__device__ __forceinline__ uint ac_token_order(CudaTokenStrategy s, uint c) {
  return c == 0 ? s.order0 : c == 1 ? s.order1 : s.order2;
}
__device__ __forceinline__ uint ac_token_signed(int value) {
  return (uint(value) << 1) ^ uint(value >> 31);
}

__global__ void gjxl_ac_token_metadata(const int* coefficients,
                                       const CudaTokenAnchor* anchors,
                                       const CudaTokenGroup* groups,
                                       const CudaTokenStrategy* strategies,
                                       const uint* orders,
                                       CudaTokenMetadata* metadata, uchar* maps,
                                       uint tasks, uint unused) {
  const uint lane = threadIdx.x & 31;
  const uint task = blockIdx.x * 4 + threadIdx.x / 32;
  if (task >= tasks) return;
  const CudaTokenAnchor a = anchors[task / 3];
  const CudaTokenStrategy s = strategies[(a.packed >> 10) & 31u];
  const uint channel = ac_token_channel(task),
             order = ac_token_order(s, channel);
  uint count = 0, last = 0;
  for (uint j = s.covered + lane; j < s.count; j += 32) {
    const int value =
        coefficients[a.source + channel * a.channel_stride + orders[order + j]];
    count += value != 0;
    if (value != 0) last = j;
  }
  count = WarpSum(count);
  last = WarpMax(last);
  if (lane == 0)
    metadata[task] = {count, last, count == 0 ? 1u : last - s.covered + 2u, 0u};
  const CudaTokenGroup g = groups[a.group];
  const uint x = a.packed & 31u, y = (a.packed >> 5) & 31u;
  if (lane < s.covered)
    maps[g.map + channel * g.width * g.height + (y + lane / s.width) * g.width +
         x + lane % s.width] = uchar((count + s.covered - 1) / s.covered);
}

__global__ void gjxl_ac_token_offsets(CudaTokenGroup* groups,
                                      CudaTokenMetadata* metadata,
                                      uint unused) {
  const uint group = blockIdx.x, lane = threadIdx.x;
  const CudaTokenGroup g = groups[group];
  uint offset = 0;
  for (uint first = 0; first < 3 * g.anchors; first += 32) {
    const uint local = first + lane, task = 3 * g.first + local;
    const uint count = local < 3 * g.anchors ? metadata[task].count : 0;
    const uint prefix = WarpPrefix(count);
    if (local < 3 * g.anchors) metadata[task].output = offset + prefix;
    offset += WarpSum(count);
  }
  if (lane == 0) groups[group].tokens = offset;
}

__global__ void gjxl_ac_token_group_offsets(CudaTokenGroup* groups,
                                            uint* control, uint group_count,
                                            uint unused) {
  const uint lane = threadIdx.x;
  uint offset = 0;
  for (uint first = 0; first < group_count; first += 32) {
    const uint group = first + lane;
    const uint count = group < group_count ? groups[group].tokens : 0;
    const uint prefix = WarpPrefix(count);
    if (group < group_count) groups[group].output = offset + prefix;
    offset += WarpSum(count);
  }
  const uint end = 4 + control[1] - 1 + 3 * 13 * control[1];
  if (lane == 0) control[end + 2] = offset;
}

__device__ __forceinline__ void ac_token_population(uint value, uint context,
                                                    uint shard, uint contexts,
                                                    uint32_t* histogram) {
  uint symbol = value;
  if (value >= 16) {
    const uint exponent = 31 - __clz(value);
    symbol = 16 + ((exponent - 4) << 2) +
             ((value - (1u << exponent)) >> (exponent - 2));
  }
  atomicAdd(histogram + (shard * contexts + context) * 128 + symbol, 1u);
}

__global__ void gjxl_ac_token_emit(
    const int* coefficients, const CudaTokenAnchor* anchors,
    const CudaTokenGroup* groups, const CudaTokenStrategy* strategies,
    const uint* orders, const CudaTokenMetadata* metadata, const uchar* maps,
    uint* values, ushort* contexts_out, uint32_t* histogram,
    const uint* control, uint unused) {
  const uint lane = threadIdx.x & 31;
  const uint task = blockIdx.x * 4 + threadIdx.x / 32;
  if (task >= control[3]) return;
  const uint control_end = 4 + control[1] - 1 + 3 * 13 * control[1];
  // A speculative compact output is published only if the whole stream fits.
  if (control[control_end + 3] &&
      control[control_end + 2] > control[control_end + 1])
    return;
  const CudaTokenAnchor a = anchors[task / 3];
  const uint strategy = (a.packed >> 10) & 31u;
  if (strategy == 0 && control[control_end + 4]) return;
  const CudaTokenStrategy s = strategies[strategy];
  const CudaTokenGroup g = groups[a.group];
  const CudaTokenMetadata m = metadata[task];
  const uint channel = ac_token_channel(task),
             order = ac_token_order(s, channel);
  const uint x = a.packed & 31u, y = (a.packed >> 5) & 31u;
  const uint num_contexts = control[0], segments = control[1],
             shards = control[2];
  const uint raw_quant = ((a.packed >> 15) & 255u) + 1u;
  uint segment = 0;
  while (segment + 1 < segments && raw_quant > control[4 + segment]) ++segment;
  const uint family = strategy == 0                    ? 0
                      : strategy == 4                  ? 2
                      : strategy == 5                  ? 3
                      : strategy == 6 || strategy == 7 ? 4
                                                       : 6;
  const uint channel_row = channel < 2 ? channel ^ 1u : 2u;
  const uint block_context =
      control[4 + segments - 1 + (channel_row * 13 + family) * segments +
              segment];
  const uint shard = (task / 3) & (shards - 1);
  if (lane == 0) {
    const uint base = g.map + channel * g.width * g.height;
    const uint prediction =
        x == 0   ? (y == 0 ? 32u : uint(maps[base + (y - 1) * g.width]))
        : y == 0 ? uint(maps[base + x - 1])
                 : (uint(maps[base + (y - 1) * g.width + x]) +
                    uint(maps[base + y * g.width + x - 1]) + 1) /
                       2;
    const uint bucket = prediction < 8     ? prediction
                        : prediction >= 64 ? 36
                                           : 4 + prediction / 2;
    const uint context = bucket * num_contexts + block_context;
    values[g.output + m.output] = m.nonzeros;
    contexts_out[g.output + m.output] = ushort(context);
    if (control[4 + segments - 1 + 3 * 13 * segments])
      ac_token_population(m.nonzeros, context, shard, num_contexts * 495,
                          histogram);
  }
  uint consumed = 0, previous = m.nonzeros > s.count / 16 ? 0u : 1u;
  for (uint first = s.covered; m.nonzeros != 0 && first <= m.last;
       first += 32) {
    const uint j = first + lane;
    const bool active = j <= m.last;
    const int coefficient =
        active ? coefficients[a.source + channel * a.channel_stride +
                              orders[order + j]]
               : 0;
    const uint nz = uint(coefficient != 0);
    const uint before = WarpPrefix(nz);
    const uint remaining = m.nonzeros - consumed - before;
    const uint previous_lane = __shfl_up_sync(0xffffffffu, nz, 1);
    const uint prev = lane == 0 ? previous : previous_lane;
    if (active) {
      const uint local =
          (kCoefficientNonzeroContext[(remaining + s.covered - 1) >>
                                      s.log2covered] +
           kCoefficientFrequencyContext[j >> s.log2covered]) *
              2 +
          prev;
      const uint context = num_contexts * 37 + 458 * block_context + local;
      const uint value = ac_token_signed(coefficient);
      values[g.output + m.output + 1 + j - s.covered] = value;
      contexts_out[g.output + m.output + 1 + j - s.covered] = ushort(context);
      if (control[4 + segments - 1 + 3 * 13 * segments])
        ac_token_population(value, context, shard, num_contexts * 495,
                            histogram);
    }
    consumed += WarpSum(nz);
    previous = __shfl_sync(0xffffffffu, nz, 31);
  }
}

// DCT8 often emits only a few coefficients. One lane per channel/transform
// avoids reserving 32 lanes for that short serial state recurrence.
__global__ void gjxl_ac_token_emit_scalar_dct8(
    const int* coefficients, const CudaTokenAnchor* anchors,
    const CudaTokenGroup* groups, const CudaTokenStrategy* strategies,
    const uint* orders, const CudaTokenMetadata* metadata, const uchar* maps,
    uint* values, ushort* contexts_out, uint32_t* histogram,
    const uint* control, uint unused) {
  const uint task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= control[3]) return;
  const uint end = 4 + control[1] - 1 + 3 * 13 * control[1];
  if (control[end + 3] && control[end + 2] > control[end + 1]) return;
  const CudaTokenAnchor a = anchors[task / 3];
  if (((a.packed >> 10) & 31u) != 0) return;
  const CudaTokenStrategy s = strategies[0];
  const CudaTokenGroup g = groups[a.group];
  const CudaTokenMetadata m = metadata[task];
  const uint channel = ac_token_channel(task),
             order = ac_token_order(s, channel);
  const uint x = a.packed & 31u, y = (a.packed >> 5) & 31u;
  const uint num_contexts = control[0], segments = control[1],
             shards = control[2];
  const uint raw_quant = ((a.packed >> 15) & 255u) + 1u;
  uint segment = 0;
  while (segment + 1 < segments && raw_quant > control[4 + segment]) ++segment;
  const uint channel_row = channel < 2 ? channel ^ 1u : 2u;
  const uint block_context =
      control[4 + segments - 1 + (channel_row * 13) * segments + segment];
  const uint shard = (task / 3) & (shards - 1);
  const uint base = g.map + channel * g.width * g.height;
  const uint prediction =
      x == 0   ? (y == 0 ? 32u : uint(maps[base + (y - 1) * g.width]))
      : y == 0 ? uint(maps[base + x - 1])
               : (uint(maps[base + (y - 1) * g.width + x]) +
                  uint(maps[base + y * g.width + x - 1]) + 1) /
                     2;
  const uint bucket = prediction < 8     ? prediction
                      : prediction >= 64 ? 36
                                         : 4 + prediction / 2;
  const uint context = bucket * num_contexts + block_context;
  const uint output = g.output + m.output;
  values[output] = m.nonzeros;
  contexts_out[output] = ushort(context);
  if (control[end])
    ac_token_population(m.nonzeros, context, shard, num_contexts * 495,
                        histogram);
  uint remaining = m.nonzeros, previous = remaining > 4 ? 0u : 1u;
  for (uint j = 1; remaining != 0 && j <= m.last; ++j) {
    const int coefficient =
        coefficients[a.source + channel * a.channel_stride + orders[order + j]];
    const uint local = (kCoefficientNonzeroContext[remaining] +
                        kCoefficientFrequencyContext[j]) *
                           2 +
                       previous;
    const uint coefficient_context =
        num_contexts * 37 + 458 * block_context + local;
    const uint value = ac_token_signed(coefficient);
    values[output + j] = value;
    contexts_out[output + j] = ushort(coefficient_context);
    if (control[end])
      ac_token_population(value, coefficient_context, shard, num_contexts * 495,
                          histogram);
    previous = uint(coefficient != 0);
    remaining -= previous;
  }
}

}  // namespace
cudaError_t LaunchCudaAcTokenEmit(CudaTokenPointers p, uint32_t tasks,
                                  bool scalar, cudaStream_t stream) {
  if (!tasks) return cudaSuccess;
  if (tasks > uint32_t(INT32_MAX) || !p.coefficients || !p.anchors ||
      !p.groups || !p.strategies || !p.orders || !p.metadata || !p.maps ||
      !p.values || !p.contexts || !p.histogram || !p.control)
    return cudaErrorInvalidValue;
  gjxl_ac_token_emit<<<(tasks + 3) / 4, 128, 0, stream>>>(
      p.coefficients, p.anchors, p.groups, p.strategies, p.orders, p.metadata,
      p.maps, p.values, p.contexts, p.histogram, p.control, 0);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  if (scalar) {
    gjxl_ac_token_emit_scalar_dct8<<<(tasks + 127) / 128, 128, 0, stream>>>(
        p.coefficients, p.anchors, p.groups, p.strategies, p.orders, p.metadata,
        p.maps, p.values, p.contexts, p.histogram, p.control, 0);
    error = cudaGetLastError();
  }
  return error;
}
cudaError_t LaunchCudaAcTokenization(CudaTokenPointers p, uint32_t tasks,
                                     uint32_t groups, bool scalar,
                                     cudaStream_t stream) {
  if (!tasks && !groups) return cudaSuccess;
  if (!tasks || !groups || tasks > uint32_t(INT32_MAX) ||
      groups > uint32_t(INT32_MAX) || !p.coefficients || !p.anchors ||
      !p.groups || !p.strategies || !p.orders || !p.metadata || !p.maps ||
      !p.values || !p.contexts || !p.histogram || !p.control)
    return cudaErrorInvalidValue;
  gjxl_ac_token_metadata<<<(tasks + 3) / 4, 128, 0, stream>>>(
      p.coefficients, p.anchors, p.groups, p.strategies, p.orders, p.metadata,
      p.maps, tasks, 0);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  gjxl_ac_token_offsets<<<groups, 32, 0, stream>>>(p.groups, p.metadata, 0);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  gjxl_ac_token_group_offsets<<<1, 32, 0, stream>>>(p.groups, p.control, groups,
                                                    0);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  return LaunchCudaAcTokenEmit(p, tasks, scalar, stream);
}
}  // namespace gjxl::cuda_internal
