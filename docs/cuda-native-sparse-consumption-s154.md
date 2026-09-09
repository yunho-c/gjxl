# Native sparse AC consumption feasibility (S154)

## Decision and scope

Direct sparse consumption is correct on this qualification set and worth
testing with a GPU producer. It is not an unconditional CPU improvement:
large compact int8 cases improve substantially in hot group replay, while
flower500 and several int32 cases regress. No production change is promoted.
The retained runtime remains S152 (`2118ccb`); this study starts from S153
(`f16637a`). Compact ownership remains opt-in.

This implements the first part of the
[S153 handoff investigation](cuda-resident-profile-s153.md#selected-next-investigation).
It does not implement GPU compression, sparse device-to-host transfer, a
sparse frame owner, or a new public API. The diagnostic full encode still
downloads the existing dense coefficients and constructs sparse input on
the CPU. That construction is for qualification, not a proposed production
pipeline or a measured acceleration. No compatibility layer is added.

## Representation and direct consumer

Each group/channel has a logical coefficient length, one 64-bit mask and
one 32-bit payload offset per 64 coefficients, and ordered signed nonzero
values in the original int8/int16/int32 width. The prototype validates exact
prefix offsets, payload counts, nonzero payload entries, header sizes, and
unused tail bits. Its group/channel limit is 65,536 coefficients.

A sparse span supplies subspans, zero-or-ranked-value lookup, and range
nonzero counting. Lookup tests the mask and uses the payload offset plus
the population count of lower bits. Counting uses masks and then removes
the original transform's LLF rectangle. The existing direct tokenizer is
instantiated on this view; its anchor walk, coefficient order, predictor-map
updates, token/context generation, and population collection are unchanged.
The sparse candidate never reconstructs a dense coefficient array.

An isolated generated copy of `src/codestream/ac_group.cpp` replaces only
the diagnostic executable's per-group entry. It runs the original dense
tokenizer, builds and checks the sparse representation, and compares every
token value, context, and population field. Crucially, it returns the
**sparse-generated tokens** to the actual serializer. The original dense
tokens are also saved as independent replay oracles. These are semantic
field comparisons, not structure-padding comparisons.

Four freshly compiled Release/host-ASAN wide/compact executables link the
frozen S152 libraries. All four contain the same ten GPU-module hashes as
the S152 production CLI. Two separately built CPU fixture executables
exercise the same tokenizer templates. There are no new CUDA bodies.

## Current coefficient census

The six frozen S153 cases use distance 1.2, effort 7, automatic CPU threads,
fully resident CUDA, and final score disabled. Larger flower/Keong inputs
are exact nearest-replicated photographs, not independent native-resolution
photographs. Two 65x67 S108 stress cases use distance 0.01, effort 7 and
exercise compact's int32 fallback. Current captures are newly generated;
historical sparse captures are not substituted for them.

The table reports active compact coefficients and logical sparse headers
plus values, in bytes. It excludes file metadata, golden tokens, allocator
capacity, GPU scratch, and any future transport framing.

| Case | Compact width | Nonzero fraction | Dense bytes | Sparse bytes |
|---|---:|---:|---:|---:|
| flower500 | int16 | 5.235% | 1,524,096 | 222,676 |
| padded HD | int8 | 1.140% | 6,220,800 | 1,237,307 |
| flower2000 | int8 | 1.340% | 12,000,000 | 2,410,785 |
| padded 4K | int8 | 1.119% | 24,883,200 | 4,944,133 |
| flower3200x2160 | int8 | 1.390% | 20,736,000 | 4,176,290 |
| keong3839x2159 | int16 | 2.610% | 49,766,400 | 5,964,348 |
| amplitude-4096 checker | int32 | 11.928% | 62,208 | 10,336 |
| amplitude-16777216 ramp | int32 | 1.993% | 62,208 | 4,156 |

Wide int32 payloads for the two 4K cases are 5,779,732 and 7,263,096 bytes,
each versus 99,532,800 active dense bytes. These are prospective byte
savings, not measured transfer savings. Current fixed headers cost 12 bytes
per 64 logical coefficients; they dominate the int8 sparse payload.

An independent binary parser checks all 994 captures, recomputes their
census, and hashes metadata, masks, offsets, int32-normalized payloads,
and oracle streams. All eight wide/compact pairs are logically identical.
All current captured LLF entries are zero; synthetic fixtures separately
exercise LLF exclusion. Padded 4K has 278,533 AC nonzeros but 568,099 ordered
coefficient tokens, including 289,566 zeros before the final nonzero.
Keong has 649,374 nonzeros and 1,420,455 coefficient tokens, including
771,081 zeros. Sparse storage does not eliminate required zero tokens or
their context calculations.

## Qualification

All 104 recorded jobs are terminal and accepted: build/native audits,
32 GPU-facing capture jobs, the CPU fixture build, 34 CPU qualification
jobs, 32 CPU timing jobs, and the three campaign orchestrators.

- 48 complete encodes match frozen codestream bytes and expected widths.
  Release uses two encodes per case/mode; host-ASAN uses one.
- 23,040 synthetic dense/sparse token comparisons and 8,478 sparse-format
  and range-count checks pass across Release and host-ASAN.
- Per build, fixtures cover 128 frames and 320 groups: seven uniform
  strategies plus mixed strategy layouts, eight coefficient patterns,
  three signed widths including their extrema, three block-context maps,
  natural and computed custom-order candidates, and population collection
  both off and on. Full/right/bottom/corner groups are included.
- Sparse span checks cover lengths 0 through 65,536, empty/dense/sparse
  patterns, unaligned ranges, and malformed headers, offsets, payloads,
  and tail masks.
- 7,952 captured replay comparisons pass across Release and host-ASAN,
  against the saved original dense token oracles.
- All 127,232 warm/measured timing calls also match those oracles;
  95,424 are measured and 31,808 are warmups.

Host-ASAN is not CUDA memcheck. No new CUDA sanitizer campaign is claimed
here; GPU modules are unchanged. Concurrent public encoding contexts are
not qualified by this diagnostic harness. No encode/timing observation is
discarded or retried. An initial postprocessing script failed on duplicate
Python keyword arguments before writing its report; its failed source is
preserved, the analysis is corrected, and original timing logs are reused.
No administrator/firewall blocker appeared. No clock, power, thermal,
priority, affinity, firewall, or security setting was changed.

## Balanced hot-consumer timing

Each captured group is replayed independently with four warmup rounds and
twelve measured rounds. Labels 0/2 are duplicate dense controls and 1/3
are duplicate sparse candidates. The order starts at 0,1,3,2 and rotates
by round, balancing position. Four label-specific scratches are retained;
token output vectors are newly allocated for each call. Two complete
repetitions traverse shuffled capture directories in opposite order.
All 994 captured groups participate in each repetition.

The timer includes tokenization and its scratch/output allocation. It
excludes sparse production, input transfer, capture reading, the dense
control's reconstruction, output destruction, and oracle comparison.
The candidate retains the tokenizer's existing worst-case output reserve.
This is hot single-group consumer feasibility, not cold frame processing,
parallel serializer wall time, or complete encoding throughput.

For each measured round index, the analyzer sums corresponding label
durations over groups, then compares the mean sparse labels with the mean
dense labels. It reports the median paired percentage across twelve round
indices. These are **virtual serial sums**, not observed frame intervals.
The ranges below span the two repetitions; negative means faster.

| Case | Wide int32 delta | Compact delta |
|---|---:|---:|
| flower500 | +16.01 to +16.38% | +7.32 to +7.38% |
| padded HD | -5.76 to -2.99% | -27.33 to -25.60% |
| flower2000 | +2.79 to +4.79% | -17.81 to -17.03% |
| padded 4K | -5.49 to -4.92% | -31.50 to -31.19% |
| flower3200x2160 | +3.90 to +4.40% | -17.86 to -16.97% |
| keong3839x2159 | +10.48 to +11.64% | -2.65 to -2.07% |
| amplitude-4096 checker | +1.69 to +5.42% | +5.76 to +6.34% |
| amplitude-16777216 ramp | +2.28 to +4.49% | +1.91 to +2.07% |

Compact padded 4K's paired virtual delta is -5.55 to -5.23 ms, with all
135 groups faster in both repetitions. Compact Keong's delta is -0.97 to
-0.78 ms, with only 76 to 80 of 135 groups faster. All four cross-label
comparisons have the same aggregate direction for each of the twelve main
case/mode combinations in both repetitions. Duplicate-label aggregate differences
stay within 2.81% there. Tiny stress controls have noisier duplicates and
some inconsistent cross-label directions; they are correctness coverage,
not evidence for a precise performance policy. Per-group and edge results,
round-label sums, cross-label comparisons, and duplicate deltas are retained.

Different width/case processes are not a causal comparison of width itself.
Source inspection establishes the tradeoff: masks replace a dense count
pass, but ordered lookup adds mask/rank work while retaining token creation.
The results do not separately attribute time to count, lookup, allocation,
or cache behavior, and do not prove sparsity alone is a sufficient selector.

## Next gate and evidence

Proceed to an isolated GPU producer and native sparse handoff experiment.
Measure packing, prefix/count synchronization, fresh allocation, actual
copies, consumption, and destruction together. Test dense/high-nonzero
inputs and small images before choosing a policy. Header size/layout is
still experimental. Do not add an unconditional sparse default based on
these byte counts or sum these hot deltas with S153's transfer medians.
The [S80 dense reconstruction failure](cuda-optimization-s1.md#lossless-sparse-ac-transfer-investigation-s80)
remains relevant: the next candidate must consume sparse data directly,
not hide fresh dense zero/scatter work behind a replay buffer.
The backend is not established to be maxed out.

Evidence is under `build-cuda-ninja/profiles/s154-artifacts/`.
`capture_protocol.json`, `qualification_inputs.json`, and
`timing_protocol.json` pin source, binary, and input identities before their
campaigns. The directory retains 994 captures, original logs, extracted
GPU modules, job records, `census.json`, `timing_analysis.json`, `report.json`,
source snapshots, and the final SHA-256 inventory. The verifier reproduces
the generated tokenizer and all analyses, validates campaign ordering and
oracle counts, and rechecks all 447 frozen S153 files and unchanged
production/test sources.

```powershell
python -X utf8 build-cuda-ninja/profiles/verify_s154.py --frozen
```

`freeze_s154.py` is an exclusive one-time finalizer, not a command to rerun
over an existing frozen evidence directory.
