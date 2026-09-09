// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

// Exercise the private validation boundary in its implementation translation
// unit without exposing a production test hook or widening the installed API.
#include "codestream/encoder.cpp"

#include <iostream>

namespace {
using namespace gjxl;

void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Check(const Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

struct Fixture {
  std::vector<std::vector<EntropyToken>> tokens{9};
  std::vector<std::vector<uint32_t>> values{9};
  std::vector<std::vector<uint16_t>> contexts{9};
  AcEncodingCandidate ac;
  SimpleCoefficientOrders orders;
  std::vector<EntropyToken> order_tokens{{0, 0}, {0, 1}, {0, 0}};
  EntropyCode order_code;

  Fixture(bool split, bool empty, bool custom) {
    constexpr std::array<uint32_t, 12> data{
      0, 1, 2, 3, 15, 16, 17, 255, 256, 65535, 1u << 24, UINT32_MAX};
    for (size_t g = 0; g < tokens.size(); ++g) {
      if (!empty && g % 4 != 0) {
        for (size_t i = 0; i < 1 + g * 5; ++i) {
          tokens[g].push_back({static_cast<uint32_t>((i + g) % 4),
                               data[(i * 7 + g) % data.size()]});
        }
      }
      values[g].push_back(0xDEADBEEFu);
      contexts[g].push_back(0xBEEFu);
      for (const auto t : tokens[g]) {
        values[g].push_back(t.value);
        contexts[g].push_back(static_cast<uint16_t>(t.context));
      }
      values[g].push_back(0xDEADBEEFu);
      contexts[g].push_back(0xBEEFu);
      ac.streams.push_back(split
        ? EntropyTokenStreamView::Split(
            std::span(values[g]).subspan(1, tokens[g].size()),
            std::span(contexts[g]).subspan(1, tokens[g].size()))
        : EntropyTokenStreamView::Interleaved(tokens[g]));
    }
    ac.custom_order = custom;
    orders.used_order_mask = custom ? 1 : 0;
    const std::array streams{EntropyTokenStreamView::Interleaved(order_tokens)};
    Check(OptimizeEntropyCode(streams, {.context_count = 1}, &order_code));
  }
};

Status Legacy(const Fixture& fixture, const EntropyCode& code,
              std::vector<BitWriter>* output, uint64_t* bits) {
  std::vector<BitWriter> candidate(1 + fixture.ac.streams.size());
  const bool custom = fixture.ac.custom_order;
  Status status = WriteSimpleAcGlobal(
    fixture.ac.streams.size(), fixture.orders.used_order_mask,
    custom ? std::span(fixture.order_tokens) : std::span<const EntropyToken>{},
    custom ? &fixture.order_code : nullptr, code, &candidate[0]);
  if (!status.ok()) return status;
  uint64_t total = 0;
  for (size_t i = 0; i < fixture.ac.streams.size(); ++i) {
    status = WriteTokenStream(fixture.ac.streams[i], code, &candidate[i + 1]);
    if (!status.ok()) return status;
    total += candidate[i + 1].bits_written();
  }
  *output = std::move(candidate);
  *bits = total;
  return Status::Ok();
}

void Same(const std::vector<BitWriter>& a, const std::vector<BitWriter>& b) {
  Check(a.size() == b.size(), "Section count changed");
  for (size_t i = 0; i < a.size(); ++i) {
    Check(a[i].bits_written() == b[i].bits_written() &&
          std::ranges::equal(a[i].padded_bytes(), b[i].padded_bytes()),
          "Section bits differ from fully validating oracle");
  }
}

void Compare(const Fixture& fixture, const EntropyCode& code, size_t budget,
             bool expected_success) {
  const auto before = code;
  std::vector<BitWriter> expected(1), actual(1);
  Check(expected[0].WriteBits(3, 5));
  Check(actual[0].WriteBits(3, 5));
  uint64_t expected_bits = 123, actual_bits = 123;
  const Status reference = Legacy(fixture, code, &expected, &expected_bits);
  thread_budget_internal::CpuParticipantTracker tracker;
  const thread_budget_internal::EncodeScope scope(budget, &tracker);
  codestream_internal::SectionWritingWorkProfile profile;
  const Status status = WriteAcSections(
    fixture.ac, code, fixture.orders, fixture.order_tokens,
    &fixture.order_code, &actual, &actual_bits, &profile);
  Check(reference.ok() == expected_success && status.ok() == expected_success,
        "Unexpected section status");
  Check(status.code() == reference.code(), "Section error category changed");
  Check(actual_bits == expected_bits && code == before,
        "Token count or borrowed model changed");
  Same(actual, expected);
  if (!expected_success) {
    Check(actual.size() == 1 && actual[0].bits_written() == 3 &&
          actual[0].padded_bytes()[0] == 5 && actual_bits == 123,
          "Failed section batch was not atomic");
  }
  if (budget) Check(tracker.peak() <= budget, "Participant budget exceeded");
}

EntropyCode Model(const Fixture& fixture, unsigned mode, bool mapped) {
  constexpr std::array<uint8_t, 4> map{0, 1, 0, 1};
  const EntropyCodeOptions options{
    .context_count = 4,
    .initial_context_map = mapped ? std::span(map) : std::span<const uint8_t>{},
    .initial_histogram_count = mapped ? 2u : 0u};
  EntropyCode code;
  if (mode == 0) Check(OptimizeEntropyCode(fixture.ac.streams, options, &code));
  else Check(codestream_internal::OptimizeDirectAnsEntropyCode(
    fixture.ac.streams, options,
    mode == 1 ? codestream_internal::DirectAnsEntropyMode::kBalanced
              : codestream_internal::DirectAnsEntropyMode::kHighDensity,
    &code));
  return code;
}

}  // namespace

int main() {
  try {
    size_t valid = 0, invalid_models = 0, invalid_tokens = 0;
    for (bool split : {false, true}) for (bool empty : {false, true})
      for (bool custom : {false, true}) for (bool mapped : {false, true})
        for (unsigned mode : {0u, 1u, 2u}) {
          Fixture fixture(split, empty, custom);
          const auto code = Model(fixture, mode, mapped);
          for (size_t budget : std::array<size_t, 4>{0, 1, 2, 8}) {
            Compare(fixture, code, budget, true);
            ++valid;
            for (unsigned fault = 0; fault < (mode ? 9u : 4u); ++fault) {
              auto broken = code;
              switch (fault) {
                case 0: broken.context_count = 0; break;
                case 1: broken.context_map[0] = 255; break;
                case 2: broken.uint_configs[0] = {16, 0, 0}; break;
                case 3: broken.uint_configs.clear(); break;
                case 4: broken.ans_log_alpha_size = 4; break;
                case 5: broken.ans_histograms.clear(); break;
                case 6: broken.ans_histograms[0].reverse_maps.push_back({}); break;
                case 7: broken.ans_histograms[0].reciprocal_frequencies.push_back(0); break;
                case 8: broken.ans_histograms[0].method = 13; break;
              }
              Compare(fixture, broken, budget, false);
              ++invalid_models;
            }
            if (!empty) {
              // The bad token is in the last populated group, after both
              // populated and empty earlier groups. Roll back the entire batch.
              auto& last = fixture.tokens[7].back();
              const auto old_context = last.context;
              last.context = code.context_count;
              auto& split_context = fixture.contexts[7][fixture.tokens[7].size()];
              const auto old_split = split_context;
              split_context = static_cast<uint16_t>(code.context_count);
              Compare(fixture, code, budget, false);
              ++invalid_tokens;
              last.context = old_context;
              split_context = old_split;
            }
          }
        }
    Fixture shared(true, false, true);
    const auto code = Model(shared, 1, true);
    std::array<std::exception_ptr, 4> errors;
    std::array<std::thread, 4> workers;
    for (size_t i = 0; i < workers.size(); ++i) workers[i] = std::thread([&, i] {
      try {
        for (size_t repeat = 0; repeat < 16; ++repeat)
          Compare(shared, code, std::array<size_t, 4>{0, 1, 2, 8}[i], true);
      } catch (...) { errors[i] = std::current_exception(); }
    });
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    std::cout << "AC section validation PASS valid=" << valid
              << " invalid_models=" << invalid_models
              << " invalid_tokens=" << invalid_tokens << " concurrent=64\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
