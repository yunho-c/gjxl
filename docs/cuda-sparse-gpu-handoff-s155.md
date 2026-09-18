# Typed GPU sparse handoff and direct consumption (S155)

## Decision

The sparse transfer benefit survives GPU packing, extra synchronization,
fresh host storage, and parallel tokenization on the six main captured
inputs. The 256-thread packer is the preferred candidate for integration.
Tiny stress inputs regress, and large dense-input performance remains an
open gate. Do not enable an unconditional sparse policy.

This advances the [S154 consumer experiment](cuda-native-sparse-consumption-s154.md)
from hot CPU group replay to a fresh AC transfer/tokenization boundary.
It starts at `7a1f969`; production remains S152 (`2118ccb`). No production
source, public API, compact default, or frame-owner implementation changes.
This is not a complete-encoder speedup claim.

## Boundary and representation

The current resident path has already selected int8/int16/int32 storage
before `AssembleTypedFrame` downloads its dense AC runs. S155 starts at
that boundary: typed final coefficients are resident on the GPU, and both
representations use the same input. The existing upstream AC group packing,
compact width detection/narrowing, image preparation, quantization, and
scoring are outside this experiment, equally for control and candidate.

The diagnostic reconstructs the immutable typed GPU input once from the
994 S154 captures, before timing. The original dense tokenizer's captured
tokens, contexts, and population fields remain the oracle. Eight cases in
wide/compact modes are used; compact widths are unchanged from S154.
Larger flower/Keong fixtures are nearest-replicated photographic inputs,
not independent native-resolution photographs.

The GPU producer processes a flat typed coefficient array. Warp ballots
describe zeros; each tile scans its warp counts and obtains a disjoint
payload interval with one atomic addition when nonempty. It writes one
64-bit mask and one absolute 32-bit payload offset per 64 coefficients,
plus typed nonzero values. Both 256- and 1024-thread tiles are instantiated
for all three signed widths. Partial tiles and partial final mask words
are supported.

Unlike S154's CPU-built representation, tile payload intervals may appear
in any completion order. Within each 64-value word the values remain in
logical order. The native sparse span uses the absolute offset and mask
rank; it does not depend on tile completion order or reconstruct dense
coefficients. Its range nonzero count replaces the dense count pass while
the original tokenizer retains LLF exclusion, coefficient scans, predictor
updates, token ordering, contexts, and population collection.

The diagnostic builds a generated copy of the current direct tokenizer
with the sparse-count overload, and copies `RunParallelSections` from the
current encoder. Automatic scheduling has the encoder's eight-worker cap,
atomic group distribution, and per-worker tokenization scratch. Group
metadata is rebased to local coordinates for replay, as in S154.

The control allocates fresh fixed-capacity host group rows, initializes
unused tails, and uses the current coalesced AC readback layout: contiguous
runs use `cudaMemcpyAsync`; narrow rows use `cudaMemcpy2DAsync`, followed by
stream synchronization. The candidate allocates fresh host headers and
fresh GPU headers/count storage, clears the count, packs, downloads
headers/count, synchronizes, validates header bounds/counts, allocates the
exact host payload, downloads it, and synchronizes again. It frees its
extra device headers inside the measured boundary.

Sparse values use a preallocated device workspace of four bytes per input
coefficient. This models reuse of the other existing coefficient allocation
after the upstream consumers finish: the resident code already reuses its
quantized/reconstruction allocations at this boundary. The experiment does
not charge either side for the already-resident input or that workspace.
Actual aliasing/lifetime integration is not implemented or proven here;
the benchmark keeps the input immutable. Extra sparse header allocation
is charged rather than amortized across the replay.

## What is timed

`ready_ns` is one continuous interval from fresh host allocation through
transfer, native parallel tokenization, and coefficient/header/scratch
destruction. It includes token-output allocation but ends with the tokens
still alive for the next consumer. Token comparison is outside timing.
Their subsequent destruction is measured separately as `release_ns`.
The reported `accounted_ns` sum includes both intervals, but is not a
continuous wall-clock interval because oracle checking lies between them.

`transfer_ns` includes allocation, packing, copies, synchronization, and
header bounds validation; it is not a GPU-only time. `token_ns` includes
natural-order preparation, worker creation/join, and direct tokenization.
The residual includes cleanup. Full coefficient/coverage checks run only
in qualification, so qualification timings are not performance evidence.

Frame metadata transfer/assembly, frame validation, coefficient-order or
block-context selection, entropy coding, final serialization, and whole
prepared-context allocation/destruction remain outside this boundary.
Metadata/order inputs are reused. Allocation contents are fresh each call,
but this does not imply a cold allocator, cold caches, or first-ever GPU
use. Complete encoding must still be measured after integration.

## Qualification

All 156 GPU-facing jobs pass: 86 qualification jobs, 64 ordinary timing
processes, and six traces. There are 163 recorded jobs including build,
native-audit, and campaign orchestration; 161 are accepted. Two failed
build launches are preserved, not counted as qualification successes.

- Six fixture runs cover 3,888 GPU packing cases, across Release,
  host-ASAN, CUDA memcheck, initcheck, racecheck, and synccheck.
- Each fixture run covers 648 combinations: three widths, both tiles,
  eighteen lengths from zero to 196,611, and six zero/dense/sparse/extreme
  patterns. It checks input immutability, guard regions, exact coefficient
  lookup, complete non-overlapping payload coverage, unwritten output tail,
  unaligned range counts, and three invalid dispatch requests.
- Captured-frame qualification checks every transferred coefficient and
  dense tail or sparse payload coverage, then checks 22,400 group token
  results against the saved original dense oracles.
- All 20 CUDA sanitizer jobs pass. Memcheck reports zero leaked bytes;
  racecheck reports zero hazards/errors/warnings. Host-ASAN is a separate
  instrumented harness check, not a substitute for CUDA sanitizers.
- All 4,096 ordinary frame-boundary calls pass their 254,464 group token
  comparisons. The six traces add 384 calls and 51,840 group comparisons.
- The two host executables contain one identical GPU module with exactly
  six kernels. Native resources report no stack, local storage, or spills.
  The 256-thread int8/int16/int32 variants use 30/32/28 registers and 68
  shared bytes; all 1024-thread variants use 40 registers and 260 shared
  bytes.

There are **no full encodes in S155**. S154's byte-exact full encodes anchor
the captured token oracles, but do not establish byte-exact integration of
a new GPU-to-frame owner. Likewise, concurrent public encoding contexts
and non-default serializer search policies are not qualified here.

## Balanced boundary results

Each process has four warmup rounds and twelve measured rounds. Labels
0/2 are duplicate dense controls and 1/3 duplicate sparse candidates.
The order 0,1,3,2 rotates by round. Case/mode/tile process order is shuffled
and reversed for the second repetition. Every call checks its token oracle
after timing. No encode-boundary or trace observation is discarded or retried.

The table gives ranges across two repetitions of the median paired
`ready_ns` percentage difference. Each round compares the mean of the two
sparse labels with the mean of the two dense labels; negative is faster.
These are measured parallel boundary intervals, not S154's virtual group
sums and not whole encoding.

| Case | Width | 256-thread delta | 1024-thread delta |
|---|---:|---:|---:|
| flower500 wide | int32 | -16.03 to -12.56% | -15.50 to -4.00% |
| flower500 compact | int16 | -28.05 to -22.30% | -23.45 to -22.71% |
| padded HD wide | int32 | -71.19 to -70.52% | -68.79 to -68.61% |
| padded HD compact | int8 | -38.65 to -36.66% | -34.75 to -32.77% |
| flower2000 wide | int32 | -68.84 to -68.17% | -67.56 to -66.79% |
| flower2000 compact | int8 | -37.28 to -36.76% | -33.14 to -31.43% |
| padded 4K wide | int32 | -75.77 to -75.38% | -73.53 to -72.75% |
| padded 4K compact | int8 | -37.64 to -36.95% | -32.99 to -30.40% |
| flower3200x2160 wide | int32 | -73.53 to -73.48% | -71.88 to -71.17% |
| flower3200x2160 compact | int8 | -39.64 to -38.95% | -36.44 to -35.40% |
| Keong wide | int32 | -63.80 to -63.20% | -61.18 to -61.08% |
| Keong compact | int16 | -42.90 to -39.37% | -38.60 to -36.86% |

For 256 threads, padded 4K's wide paired saving is 20.07–20.69 ms, and its
compact saving is 4.07–4.37 ms. Compact Keong saves 8.86–9.00 ms. All four
cross-label comparisons have the same aggregate direction for all twelve
main case/mode combinations with 256 threads. Duplicate-label and phase
results are retained; they are not assumed to be zero.

Including separately measured output destruction leaves compact padded
4K's paired component-sum improvement at 32.52–32.67%, and compact Keong
at 36.60–38.49%. Those component sums are not whole-encoder predictions.

The two 65x67 int32 stress cases regress with 256 threads: approximately
20–38 microseconds, or 14–28%, across wide/compact runs. The 1024-thread
stress observations are also retained, including one noisy +68.46%
repetition. Small inputs require an explicit policy. High-density packing
is correctness-tested, but large high-density transfer/tokenization
performance is not measured yet. Six sparse main inputs cannot determine
a general density threshold.

## GPU trace attribution

Nsight Systems traces are separate from ordinary timing. Full CUDA capture
uses no CPU sampling/context-switch collection. The analyzer matches the
entire ordered copy/kernel sequence to all 64 boundary calls per trace,
checks successful runtime API returns, launch geometry/resources, exact
copy counts/bytes, and non-overlapping stream order. One initial typed H2D
upload is outside the boundary. Each sparse call has one count clear, one
packing kernel, and three D2H copies: headers, a four-byte count, and values.

| Input/width | Tile | Pack median | Sparse D2H median | Dense D2H median |
|---|---:|---:|---:|---:|
| padded 4K int32 | 256 | 0.684 ms | 1.224 ms | 27.171 ms |
| padded 4K int32 | 1024 | 1.419 ms | 0.959 ms | 17.499 ms |
| padded 4K int8 | 256 | 0.673 ms | 0.869 ms | 4.179 ms |
| padded 4K int8 | 1024 | 1.421 ms | 0.807 ms | 3.931 ms |
| Keong int16 | 256 | 0.669 ms | 1.174 ms | 9.695 ms |
| Keong int16 | 1024 | 1.457 ms | 0.985 ms | 9.195 ms |

Measured sparse wire bytes, including the count, are 5,779,736 for padded
4K int32, 4,944,137 for padded 4K int8, and 5,964,352 for Keong int16.
Their controls transfer 99,532,800, 24,883,200, and 49,766,400 bytes.
Header/payload host byte fields are logical sizes; a header allocation may
round up by four bytes, and allocator/runtime metadata is not counted.

The 256-thread kernel is consistently faster in these traces despite
more potential atomic reservations. Resources and source structure do not
by themselves establish the precise cause of the larger tile's slowdown.
The wide control's 27.17 versus 17.50 ms copy medians show meaningful
between-process variation; retain both rather than attributing it uniquely
to clocks, power, or PCIe. Separate tile processes are not a direct paired
256-versus-1024 experiment. CUDA waits overlap GPU activity and must not be
added to it as independent work. No power/clock constancy is claimed.

## Next gate and evidence

Integrate the candidate into a native frame-owned sparse representation,
including validation, order-population use, other coefficient consumers,
and lifetimes. Measure large high-density cases and choose a small/dense
input policy before promotion. Then require complete resident encodes,
codestream/reconstruction correctness, fresh-context and repeated-context
timings, sanitizer coverage, and concurrency tests. No dense reconstruction
adapter or unconditional sparse default is justified by this boundary test.
The backend is not established to be maxed out.

Evidence is in `build-cuda-ninja/profiles/s155-artifacts/`: pinned protocols,
captures by reference, binaries, six-kernel module extracts/resources/SASS,
original logs, traces/SQLite/stdout, `analysis.json`, and `handoff.json`.
The first build lacked `<system_error>`; the next launch had a command-line
typo. Failed source/logs are preserved. Two postprocessing assertions were
corrected without rerunning traces: the demangler's template syntax and
Nsight's nonzero aggregate local-memory field, despite zero per-thread
local storage/native spills. Their failed sources are preserved too.

Importing an older analysis reset the shared runner's journal directory,
placing twelve new trace log/JSON files beside frozen S154 evidence. Only
those newly owned files were moved to S155, after checking exact targets,
job identities, hashes, and absence from the old manifest. The recovery
record preserves all hashes; full frozen S154 verification passes again.
Existing frozen artifacts were not overwritten. New verification explicitly
restores the runner directory after historical analysis imports.

No firewall/admin blocker appeared. No clock, power, thermal, affinity,
priority, firewall, or persistent security setting was changed. The final
source snapshot and artifact inventory are frozen after report validation.

```powershell
python -X utf8 build-cuda-ninja/profiles/verify_s155.py --frozen
```

`freeze_s155.py` is a one-time exclusive finalizer, not a command to rerun
over the frozen directory.
