// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/entropy_storage_plan.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codestream/ans_internal.h"
#include "codestream/huffman.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
using enum EntropyStoragePolicy;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Ok(const Status &status) {
  if (!status.ok())
    std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                   s.total.pending_count == 0 && s.open_reservations == 0 &&
                   s.waiting_requests == 0,
               "Entropy plan test leaked a resource charge");
}

struct Backings {
  size_t bytes = 0;
  size_t count = 0;
  template <typename T> void Add(const Storage<T> &v) {
    bytes += v.capacity() * sizeof(T);
    count += v.capacity() != 0;
  }
  void Add(const EntropyCode &code) {
    Add(code.context_map);
    Add(code.uint_configs);
    Add(code.prefix_codes);
    Add(code.ans_histograms);
    for (const auto &h : code.ans_histograms) {
      Add(h.frequencies);
      Add(h.reverse_maps);
      Add(h.reciprocal_frequencies);
      for (const auto &reverse : h.reverse_maps)
        Add(reverse);
    }
  }
  void Add(const PreparedEntropyClusters &p) {
    Add(p.context_map);
    Add(p.values);
    Add(p.fixed_ans_clusters);
    for (const auto &v : p.values)
      Add(v);
  }
  bool Matches(const ResourceBudget &budget, size_t bound) const {
    const auto s = budget.snapshot();
    const auto &serial =
        s.classes[static_cast<size_t>(ResourceClass::kSerializer)];
    return Check(
        bytes <= bound && bytes == s.total.live_capacity_bytes &&
            count == s.total.backing_count &&
            bytes == serial.live_capacity_bytes &&
            count == serial.backing_count && s.total.pending_count == 0,
        "Entropy output capacity/owner differs from its ledger or bound");
  }
};

struct Result {
  EntropyCode code;
  EntropyCodeCost cost;
  PreparedEntropyClusters prepared;
  PreparedAnsEntropyCode deferred;
  bool operator==(const Result &other) const {
    if (code != other.code || cost != other.cost ||
        prepared != other.prepared ||
        deferred.section_count != other.deferred.section_count ||
        deferred.candidates.size() != other.deferred.candidates.size())
      return false;
    for (size_t i = 0; i < deferred.candidates.size(); ++i) {
      const auto &a = deferred.candidates[i];
      const auto &b = other.deferred.candidates[i];
      if (a.code != b.code || a.model_bits != b.model_bits ||
          a.minimum_token_bits != b.minimum_token_bits ||
          a.survives != b.survives)
        return false;
    }
    return true;
  }
  Backings Owned() const {
    Backings b;
    b.Add(code);
    b.Add(cost.section_token_bits);
    b.Add(prepared);
    b.Add(deferred.candidates);
    for (const auto &c : deferred.candidates)
      b.Add(c.code);
    return b;
  }
};

uint32_t Random(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

bool Aggregation() {
  size_t cases = 0;
  for (size_t n : {0ul, 1ul, 255ul, 256ul, 257ul, 4095ul, 4096ul, 4097ul,
                   65536ul, 262144ul}) {
    for (size_t pattern = 0; pattern < 5; ++pattern) {
      std::vector<uint32_t> values(n);
      uint32_t random = 123;
      for (size_t i = 0; i < n; ++i) {
        switch (pattern) {
        case 0:
          values[i] = 7;
          break;
        case 1:
          values[i] = static_cast<uint32_t>(i % 65536);
          break;
        case 2:
          values[i] = UINT32_MAX - static_cast<uint32_t>(i);
          break;
        case 3:
          values[i] = Random(random);
          break;
        default:
          values[i] = i % 2 == 0 ? 65536 : UINT32_MAX;
          break;
        }
      }
      auto sorted = values;
      std::ranges::sort(sorted);
      std::vector<WeightedValue> oracle;
      for (uint32_t value : sorted) {
        if (oracle.empty() || oracle.back().value != value)
          oracle.push_back({value, 1});
        else
          ++oracle.back().count;
      }
      EntropyAggregationStoragePlan plan;
      if (!Ok(ComputeEntropyAggregationStoragePlan(n, &plan)))
        return false;
      HostStorageBound total = plan.output;
      if (!Check(total.Add(plan.scratch), "Aggregation bound sum failed"))
        return false;
      ResourceBudget budget;
      ResourceReservation job;
      if (!Ok(budget.Reserve(std::max(size_t{1}, total.peak_bytes), &job)))
        return false;
      Storage<WeightedValue> output;
      {
        ResourceContextScope context({&job, ResourceClass::kPreparation});
        if (!Ok(AggregateEntropyValues(std::span<uint32_t>(values), &output)) ||
            !Check(std::ranges::equal(output, oracle),
                   "Aggregation changed values/counts"))
          return false;
      }
      Backings b;
      b.Add(output);
      if (!b.Matches(budget, plan.output.retained_bytes) ||
          !Check(budget.snapshot().peak_backing_bytes <= total.peak_bytes,
                 "Aggregation exceeded its reservation"))
        return false;
      job.Reset();
      if (!b.Matches(budget, plan.output.retained_bytes))
        return false;
      output = {};
      // Empty assignment can retain capacity; swap explicitly releases backing.
      Storage<WeightedValue>().swap(output);
      if (!Empty(budget))
        return false;
      ++cases;
    }
  }
  std::cout << cases << " aggregation cases checked\n";
  return true;
}

Status Optimize(const EntropyOptimizationStorageOptions &o,
                std::span<const EntropyTokenStreamView> views,
                const EntropyCodeOptions &input_options,
                const EntropyCode &prefix,
                const PreparedEntropyClusters &prepared, Result *out) {
  auto *cost = o.return_cost ? &out->cost : nullptr;
  switch (o.policy) {
  case kFastPrefix:
    return OptimizeFastPrefixEntropyCode(views, input_options, &out->code,
                                         cost);
  case kPrefix:
    if (o.retain_prepared_clusters)
      return OptimizeEntropyCodeAndPrepareClusters(
          views, input_options, &out->code, cost, &out->prepared);
    return OptimizeEntropyCode(views, input_options, &out->code, cost);
  case kBalancedAns:
  case kHighDensityAns:
    return OptimizeDirectAnsEntropyCode(
        views, input_options,
        o.policy == kBalancedAns ? DirectAnsEntropyMode::kBalanced
                                 : DirectAnsEntropyMode::kHighDensity,
        &out->code, cost);
  case kAnsFromPrefix:
    if (o.borrow_prepared_clusters)
      return OptimizeAnsEntropyCodeWithPreparedClusters(views, prefix, prepared,
                                                        &out->code, cost);
    return OptimizeAnsEntropyCode(views, prefix, &out->code, cost);
  case kDeferredAnsFromPrefix:
    return PrepareAnsEntropyCodeWithPreparedClusters(views, prefix, prepared,
                                                     &out->deferred);
  }
  return Status::Internal("Test policy invalid");
}

bool ModelAndEmission(const EntropyCode &model, EntropyTokenStreamView tokens) {
  EntropyModelStoragePlan mp;
  EntropyTokenEmissionStoragePlan ep;
  const size_t clusters = model.mode == EntropyCodingMode::kPrefix
                              ? model.prefix_codes.size()
                              : model.ans_histograms.size();
  if (!Ok(ComputeEntropyModelStoragePlan(model.mode, model.context_count,
                                         clusters, &mp)) ||
      !Ok(ComputeEntropyTokenEmissionStoragePlan(model.mode, tokens.size(),
                                                 &ep)))
    return false;
  Backings model_backing;
  model_backing.Add(model);
  if (!Check(model_backing.bytes <= mp.owned.retained_bytes,
             "Actual optimizer model exceeds model-only bound"))
    return false;
  BitWriter oracle;
  if (!Ok(oracle.WriteBits(3, 5)) || !Ok(WriteEntropyCode(model, &oracle)))
    return false;
  const size_t model_bits = oracle.bits_written() - 3;
  if (!Check(model_bits <= mp.maximum_bits,
             "Serialized model exceeded bit bound"))
    return false;
  if (!Ok(WriteTokenStream(tokens, model, &oracle)) ||
      !Ok(WriteTokenStream(tokens, model, &oracle)))
    return false;
  HostStorageBound bound;
  if (!Ok(ComputeEntropyWriterStorageBound(
          3 + mp.maximum_bits + 2 * ep.maximum_bits, &bound)) ||
      !Check(bound.Add(mp.write_scratch) && bound.Add(ep.scratch),
             "Emission sum failed"))
    return false;
  ResourceBudget budget;
  ResourceReservation job;
  if (!Ok(budget.Reserve(bound.peak_bytes, &job)))
    return false;
  {
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    BitWriter writer;
    if (!Ok(writer.WriteBits(3, 5)) || !Ok(WriteEntropyCode(model, &writer)) ||
        !Ok(WriteTokenStream(tokens, model, &writer)) ||
        !Ok(WriteTokenStream(tokens, model, &writer)) ||
        !Check(writer.bits_written() == oracle.bits_written() &&
                   std::ranges::equal(writer.padded_bytes(),
                                      oracle.padded_bytes()),
               "Managed emission differs from oracle") ||
        !Check(writer.bits_written() - 3 - model_bits <= 2 * ep.maximum_bits &&
                   budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
               "Token emission exceeded bit/backing bound"))
      return false;
  }
  job.Reset();
  return Empty(budget);
}

bool OptimizationCase(size_t contexts, size_t n, size_t sections,
                      size_t pattern, bool initial_map,
                      bool failure_sweep = false) {
  std::vector<uint32_t> values(n);
  std::vector<uint16_t> token_contexts(n);
  uint32_t random = 13;
  for (size_t i = 0; i < n; ++i) {
    token_contexts[i] = static_cast<uint16_t>(i % contexts);
    values[i] =
        pattern == 0 ? 0
        : pattern == 1
            ? Random(random)
            : static_cast<uint32_t>((i / contexts) % (3 + i % contexts));
  }
  if (pattern == 3) {
    random = 13;
    for (size_t c = 0; c < contexts; ++c) {
      const uint32_t modulus = 3 + Random(random) % 30,
                     offset = Random(random) % 24;
      for (size_t i = 0; i < 160; ++i) {
        values[c * 160 + i] = offset + Random(random) % modulus;
        token_contexts[c * 160 + i] = static_cast<uint16_t>(c);
      }
    }
  }
  std::vector<EntropyTokenStreamView> views;
  for (size_t s = 0; s < sections; ++s) {
    const size_t begin = n * s / sections, end = n * (s + 1) / sections;
    views.push_back(EntropyTokenStreamView::Split(
        std::span<const uint32_t>(values).subspan(begin, end - begin),
        std::span<const uint16_t>(token_contexts).subspan(begin, end - begin)));
  }
  std::vector<uint8_t> map(initial_map ? contexts : 0);
  for (size_t i = 0; i < map.size(); ++i)
    map[i] = static_cast<uint8_t>(i % 17);
  const EntropyCodeOptions input{
      .context_count = static_cast<uint32_t>(contexts),
      .initial_context_map = map,
      .initial_histogram_count = initial_map ? 17u : 0u,
  };
  EntropyCode prefix;
  EntropyCodeCost prefix_cost;
  PreparedEntropyClusters prepared;
  if (!Ok(OptimizeEntropyCodeAndPrepareClusters(views, input, &prefix,
                                                &prefix_cost, &prepared)))
    return false;
  std::vector<PreparedFixedAnsCluster> fixed(contexts);
  for (size_t i = 0; i < n; ++i) {
    HybridUintToken token;
    if (!Ok(EncodeHybridUint(values[i], input.uint_config, &token)))
      return false;
    auto &population = fixed[token_contexts[i]];
    ++population.counts[token.symbol];
    ++population.token_count;
    population.extra_bits += token.extra_bit_count;
    population.maximum_symbol =
        std::max(population.maximum_symbol, token.symbol);
  }
  for (size_t variant = 0; variant < 8; ++variant) {
    const std::array policies{kFastPrefix,     kPrefix,
                              kPrefix,         kBalancedAns,
                              kHighDensityAns, kAnsFromPrefix,
                              kAnsFromPrefix,  kDeferredAnsFromPrefix};
    EntropyOptimizationStorageOptions o{
        .policy = policies[variant],
        .tokens = n,
        .contexts = contexts,
        .sections = sections,
        .initial_histograms = initial_map ? 17ul : 0ul,
        .return_cost = pattern != 0 || variant == 2,
        .retain_prepared_clusters = variant == 2,
        .borrow_prepared_clusters = variant >= 6,
    };
    const auto run = [&](Result *out) {
      if (variant == 3 && initial_map)
        return OptimizeDirectAnsEntropyCodeWithFixedPopulations(
            views, input, fixed, &out->code,
            o.return_cost ? &out->cost : nullptr);
      return Optimize(o, views, input, prefix, prepared, out);
    };
    EntropyOptimizationStoragePlan plan;
    Result oracle;
    if (!Ok(ComputeEntropyOptimizationStoragePlan(o, &plan)) ||
        !Ok(run(&oracle)))
      return false;
    if (pattern == 3 && (variant == 3 || variant == 4) &&
        !Check(oracle.code.ans_histograms.size() == (variant == 3 ? 9 : 8),
               "Queue fixture no longer exercises a beneficial merge"))
      return false;
    ResourceBudget budget;
    ResourceReservation job;
    if (!Ok(budget.Reserve(plan.working.peak_bytes, &job)))
      return false;
    Result output;
    {
      ResourceContextScope context({&job, ResourceClass::kPreparation});
      if (!Ok(run(&output)) ||
          !Check(output == oracle, "Optimizer parity failed"))
        return false;
    }
    if (!output.Owned().Matches(budget, plan.output.retained_bytes))
      return false;
    if (variant == 7) {
      std::vector<uint64_t> measured(sections *
                                     output.deferred.candidates.size());
      for (size_t s = 0; s < sections; ++s) {
        if (!Ok(MeasurePreparedAnsEntropyCodeSection(
                views[s], output.deferred,
                std::span<uint64_t>(measured).subspan(
                    s * output.deferred.candidates.size(),
                    output.deferred.candidates.size()))))
          return false;
      }
      if (failure_sweep) {
        const Result before = output;
        bool completed = false;
        for (size_t fail = 0; fail < 8; ++fail) {
          ResourceContextScope context({&job, ResourceClass::kPreparation});
          ArmManagedHostAllocationFailureAfterForTest(fail);
          const auto status = FinalizePreparedAnsEntropyCode(
              &output.deferred, measured, &output.code,
              o.return_cost ? &output.cost : nullptr);
          const bool injected = !ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          if (!injected) {
            if (!Ok(status))
              return false;
            completed = true;
            break;
          }
          if (!Check(status.code() == StatusCode::kOutOfMemory &&
                         output == before,
                     "Failed deferred finalization changed retained models") ||
              !output.Owned().Matches(budget, plan.output.retained_bytes))
            return false;
        }
        if (!Check(completed, "Finalization failure sweep never succeeded"))
          return false;
      } else {
        ResourceContextScope context({&job, ResourceClass::kPreparation});
        if (!Ok(FinalizePreparedAnsEntropyCode(
                &output.deferred, measured, &output.code,
                o.return_cost ? &output.cost : nullptr)))
          return false;
      }
      Result immediate;
      o.policy = kAnsFromPrefix;
      if (!Ok(run(&immediate)) ||
          !Check(output == immediate,
                 "Deferred finalization differs from immediate ANS"))
        return false;
      o.policy = kDeferredAnsFromPrefix;
    }
    if (!Check(budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes,
               "Optimizer exceeded working bound"))
      return false;
    job.Reset();
    if (!output.Owned().Matches(budget, plan.output.retained_bytes))
      return false;
    if (!views.empty() && !ModelAndEmission(output.code, views.back()))
      return false;
    output = {};
    if (!Empty(budget))
      return false;

    // Exhaust the admitted credit before the first physical allocation. The
    // fault hook must remain armed, proving rejection occurred before malloc.
    if (!Ok(budget.Reserve(1, &job)) || !Ok(job.ReduceCapacity(0)))
      return false;
    Result sentinel;
    sentinel.code.context_count = 77;
    sentinel.cost.model_bits = 19;
    output = sentinel;
    {
      ResourceContextScope context({&job, ResourceClass::kSerializer});
      ArmManagedHostAllocationFailureAfterForTest(0);
      const auto status = run(&output);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (!Check(
              status.resource_plan_exceeded() && pending && output == sentinel,
              "Underplanned entropy job escaped, allocated, or changed output"))
        return false;
    }
    job.Reset();
    if (!Empty(budget))
      return false;

    if (failure_sweep && (pattern != 3 || variant == 4)) {
      Result recovered = sentinel;
      if (!Ok(run(&recovered)))
        return false;
      bool finished = false;
      for (size_t fail = 0; fail < 16384; ++fail) {
        if (!Ok(budget.Reserve(plan.working.peak_bytes, &job)))
          return false;
        output = sentinel;
        bool injected;
        {
          ResourceContextScope context({&job, ResourceClass::kPreparation});
          ArmManagedHostAllocationFailureAfterForTest(fail);
          const auto status = run(&output);
          injected = !ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          if (injected) {
            if (!Check(status.code() == StatusCode::kOutOfMemory &&
                           !status.resource_plan_exceeded() &&
                           output == sentinel,
                       "Entropy allocation failure was not atomic"))
              return false;
          } else if (!Ok(status) ||
                     !Check(output == recovered, "Failure recovery differs")) {
            return false;
          }
        }
        output = {};
        job.Reset();
        if (!Empty(budget))
          return false;
        if (!injected) {
          std::cout << "Variant " << variant << ": " << fail
                    << " failures checked\n";
          finished = true;
          break;
        }
      }
      if (!Check(finished, "Entropy failure sweep exhausted its test limit"))
        return false;
    }
  }
  return true;
}

bool RefinementQueue() {
  // Same deterministic nine-to-eight cluster fixture as the entropy primitive
  // test. Sweep EVERY allocation position, including the beneficial merge's
  // managed queue push, rather than inferring queue coverage from a no-merge
  // case.
  return OptimizationCase(12, 12 * 160, 1, 3, false, true);
}

bool WriterFailures(const EntropyCode &model, EntropyTokenStreamView tokens) {
  const size_t clusters = model.mode == EntropyCodingMode::kPrefix
                              ? model.prefix_codes.size()
                              : model.ans_histograms.size();
  EntropyModelStoragePlan mp;
  EntropyTokenEmissionStoragePlan ep;
  if (!Ok(ComputeEntropyModelStoragePlan(model.mode, model.context_count,
                                         clusters, &mp)) ||
      !Ok(ComputeEntropyTokenEmissionStoragePlan(model.mode, tokens.size(),
                                                 &ep)))
    return false;
  for (bool model_only : {true, false}) {
    const auto write = [&](BitWriter *writer) {
      return model_only ? WriteEntropyCode(model, writer)
                        : WriteTokenStream(tokens, model, writer);
    };
    BitWriter oracle;
    if (!Ok(oracle.WriteBits(3, 5)) || !Ok(write(&oracle)))
      return false;
    HostStorageBound bound;
    if (!Ok(ComputeEntropyWriterStorageBound(
            3 + (model_only ? mp.maximum_bits : ep.maximum_bits), &bound)) ||
        !Check(bound.Add(model_only ? mp.write_scratch : ep.scratch),
               "Writer bound sum failed"))
      return false;
    bool completed = false;
    for (size_t fail = 0; fail < 1024; ++fail) {
      ResourceBudget budget;
      ResourceReservation job;
      if (!Ok(budget.Reserve(bound.peak_bytes, &job)))
        return false;
      BitWriter writer;
      // The pre-existing destination is caller owned, outside this component.
      if (!Ok(writer.WriteBits(3, 5)))
        return false;
      bool injected;
      {
        ResourceContextScope context({&job, ResourceClass::kPreparation});
        ArmManagedHostAllocationFailureAfterForTest(fail);
        const auto status = write(&writer);
        injected = !ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (injected) {
          if (!Check(status.code() == StatusCode::kOutOfMemory &&
                         !status.resource_plan_exceeded() &&
                         writer.bits_written() == 3 &&
                         writer.padded_bytes().size() == 1 &&
                         writer.padded_bytes()[0] == 5 &&
                         budget.snapshot().total.backing_count == 0,
                     "Entropy writer allocation failure was not atomic"))
            return false;
        } else if (!Ok(status) ||
                   !Check(writer.bits_written() == oracle.bits_written() &&
                              std::ranges::equal(writer.padded_bytes(),
                                                 oracle.padded_bytes()),
                          "Writer recovery differs from oracle"))
          return false;
      }
      writer = {};
      job.Reset();
      if (!Empty(budget))
        return false;
      if (!injected) {
        std::cout << "Writer " << static_cast<unsigned>(model.mode) << '/'
                  << model_only << ": " << fail << " failures checked\n";
        completed = true;
        break;
      }
    }
    if (!Check(completed, "Writer failure sweep did not finish"))
      return false;
  }
  return true;
}

bool FullAlphabetModels() {
  // Synthetic valid freshly sized models stress all 32 clusters, every reverse
  // table slot, a long context map, and a deep nondegenerate Prefix tree.
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    EntropyCode code;
    code.mode = mode;
    code.context_count = 7425;
    code.context_map.resize(code.context_count);
    for (size_t c = 0; c < code.context_map.size(); ++c)
      code.context_map[c] = c % 32;
    code.uint_configs.resize(32, mode == EntropyCodingMode::kPrefix
                                     ? kDefaultHybridUintConfig
                                     : HybridUintConfig{8, 0, 0});
    if (mode == EntropyCodingMode::kPrefix) {
      PrefixCode prefix;
      std::array<uint64_t, kPrefixAlphabetSize> counts;
      for (size_t i = 0; i < counts.size(); ++i)
        counts[i] = uint64_t{1} << (i % 16);
      if (!Ok(CreateHuffmanTree(counts, 15, prefix.depths)) ||
          !Ok(ConvertBitDepthsToSymbols(prefix.depths, prefix.bits)))
        return false;
      code.prefix_codes.resize(32, prefix);
    } else {
      code.ans_log_alpha_size = 8;
      code.ans_histograms.resize(32);
      for (auto &h : code.ans_histograms) {
        h.method = 12;
        h.frequencies.assign(256, 16);
        h.reciprocal_frequencies.assign(256, AnsFrequencyReciprocal(16));
        h.reverse_maps.resize(256);
        for (size_t s = 0; s < 256; ++s) {
          h.reverse_maps[s].resize(16);
          for (size_t i = 0; i < 16; ++i)
            h.reverse_maps[s][i] = s * 16 + i;
        }
      }
    }
    std::vector<EntropyToken> tokens(4097);
    uint32_t random = 871;
    for (auto &token : tokens) {
      token.context = Random(random) % code.context_count;
      token.value = mode == EntropyCodingMode::kPrefix ? Random(random)
                                                       : Random(random) % 256;
    }
    const auto view = EntropyTokenStreamView::Interleaved(tokens);
    if (!ModelAndEmission(code, view) || !WriterFailures(code, view))
      return false;
  }
  return true;
}

bool InvalidAndLarge() {
  constexpr size_t maximum = std::numeric_limits<size_t>::max();
  EntropyOptimizationStoragePlan sentinel{19, {23, 29}, {31, 37}},
      out = sentinel;
  for (EntropyOptimizationStorageOptions o :
       {EntropyOptimizationStorageOptions{},
        {.contexts = 1, .initial_histograms = 257},
        {.policy = kFastPrefix,
         .contexts = 1,
         .retain_prepared_clusters = true},
        {.policy = kBalancedAns,
         .contexts = 1,
         .borrow_prepared_clusters = true},
        {.policy = kDeferredAnsFromPrefix, .contexts = 1},
        {.policy = static_cast<EntropyStoragePolicy>(255), .contexts = 1},
        {.tokens = maximum, .contexts = 1},
        {.contexts = 1, .sections = maximum}}) {
    if (!Check(!ComputeEntropyOptimizationStoragePlan(o, &out).ok() &&
                   out == sentinel,
               "Invalid/overflow optimizer plan modified output"))
      return false;
  }
  EntropyTokenEmissionStoragePlan es{1, 2, {3, 4}}, e = es;
  EntropyModelStoragePlan ms{1, {2, 3}, {4, 5}}, m = ms;
  HostStorageBound bs{7, 11}, b = bs;
  size_t chunks = 19;
  if (!Check(
          !ComputeEntropyTokenEmissionStoragePlan(EntropyCodingMode::kAns,
                                                  maximum, &e)
                  .ok() &&
              e == es &&
              !ComputeEntropyTokenEmissionStoragePlan(
                   EntropyCodingMode::kPrefix, maximum, &e)
                   .ok() &&
              e == es &&
              !ComputeEntropyModelStoragePlan(EntropyCodingMode::kPrefix, 1, 33,
                                              &m)
                   .ok() &&
              m == ms && !ComputeEntropyWriterStorageBound(maximum, &b).ok() &&
              b == bs && !ComputeAnsReverseChunkCount(maximum, &chunks).ok() &&
              chunks == 19,
          "Component overflow changed output"))
    return false;
  // Huge but representable count-only plans must not allocate or iterate N/H.
  ArmManagedHostAllocationFailureAfterForTest(0);
  bool good = true;
  for (auto policy : {kFastPrefix, kPrefix, kBalancedAns, kHighDensityAns,
                      kAnsFromPrefix, kDeferredAnsFromPrefix}) {
    good &= ComputeEntropyOptimizationStoragePlan(
                {.policy = policy,
                 .tokens = size_t{1} << 32,
                 .contexts = UINT32_MAX,
                 .sections = size_t{1} << 20,
                 .borrow_prepared_clusters = policy == kDeferredAnsFromPrefix},
                &out)
                .ok();
    good &= out.output.retained_bytes <= out.working.retained_bytes &&
            out.working.retained_bytes <= out.working.peak_bytes;
  }
  good &= ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  return Check(good, "Large count-only plans failed or allocated");
}
} // namespace

int main() {
  if (!Empty(DefaultResourceBudget()) || !InvalidAndLarge() || !Aggregation() ||
      !FullAlphabetModels() || !RefinementQueue() ||
      !OptimizationCase(2, 96, 3, 2, false, true) ||
      !OptimizationCase(7, 0, 3, 0, false))
    return EXIT_FAILURE;
  size_t cases = 0;
  for (size_t contexts : {1ul, 7ul, 33ul, 257ul}) {
    for (size_t pattern = 0; pattern < 3; ++pattern) {
      for (bool initial : {false, true}) {
        if (!OptimizationCase(contexts, pattern == 0 ? 0 : 4097,
                              pattern == 0 ? 0 : 5, pattern, initial))
          return EXIT_FAILURE;
        cases += 8;
      }
    }
  }
  if (!Empty(DefaultResourceBudget()) ||
      !Check(DefaultResourceBudget().snapshot().peak_backing_bytes == 0,
             "An explicit entropy job escaped to the default domain"))
    return EXIT_FAILURE;
  std::cout
      << cases
      << " optimizer matrix cases passed; all entropy storage tests passed.\n";
  return EXIT_SUCCESS;
}
