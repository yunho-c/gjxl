# Private strategy-search cell encoding (S174)

September 10, 2026. Starting revision: `01a9b13`, branch `feat/cuda`.
Windows, RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37 Release;
scoped host ASAN uses clang-cl 22.

## Outcome

Neither candidate is promoted. A direct edge-bit representation fails
predicate equivalence in an extreme-cost intermediate search state. A
geometry-encoded cell preserves the original traversal and passes the host
and whole-encoder correctness gates, but does not establish a dependable
merge or complete-encode speedup. Runtime source remains unchanged.

This follows the frozen S173 token-count study. Reviewing S169's retained
traces led back to strategy selection, but S103/S105/S113 already distinguish
the CPU merge, preparation and metadata boundaries. S106/S112 already tested
parallel CPU tile scheduling; S135/S136 tested packed host costs and generated
candidate descriptors. This is a different, serial algorithm-representation
experiment, not another scheduling sweep. No new GPU trace was captured.
The exploratory S169 gap reader is retained but is not a new all-trace
attribution study or a basis for an additional quantified speed claim.

## First candidate: edge bits are not equivalent

The private search grid originally stores the strategy index and anchor flag
in one byte per base block. The first candidate retains those low six bits,
and records whether the cell crosses its left/top edges in bits 6/7.
Boundary queries inspect those flags rather than walking backward to an
anchor and hopping through transform footprints. The byte-cell owner,
search arithmetic, merge order and export are unchanged.

The diagnostic compares each new predicate with the actual old traversal,
using decoded strategies from the same intermediate grid. It also checks
complete outputs against a separately compiled production search. Direct
queries on 256 valid randomly tiled grids pass 480,128 comparisons.
That is insufficient: the search can temporarily contain inconsistent
coverage for extreme finite costs, and the original boundary walk has
different behavior there.

The retained counterexample is an 11 by 13 base-block grid, cost pattern 6,
seed 175672. Pattern 6 selects finite maximum or minimum positive floats.
For horizontal query `(start_x=2, y=2, end_x=6)`, the edge bits return false
while the original walk returns true. The exception occurs inside selection,
not in a post hoc final-grid comparison. No timing or whole encoder uses
this failed candidate. A standalone one-case reproducer confirms the same
predicate mismatch without preceding cases or an injected invalid leaf.
Its source, diagnostic and failure logs are retained. The expected exception
is a successful reproduction, not a correctness pass for the candidate.

Two earlier fixture attempts incorrectly required every finite-cost case to
succeed. At 4 by 3 blocks, pattern 6, both implementations instead return the
same existing internal incomplete-grid error. A diagnostic-only version
identified that overstrict fixture expectation. Version 3 retains all cases
but requires identical status/message and unchanged nonempty sentinel output
when the original fails; it still requires success for patterns 0–5. This
then exposes the real predicate mismatch above. These are separate findings;
the original extreme-cost error is not blamed on the new representation.
No arithmetic or error behavior is changed to make a performance test pass.

## Second candidate: geometry in the private byte

The second representation stores the anchor flag in bit 0, width minus one
in bits 1–2, and height minus one in bits 3–4. An inverse 16-entry table
recovers the strategy where selection/export needs it. A compile-time check
requires exact round trips for all seven policy strategies and footprints
no larger than four base blocks in either dimension.

Boundary queries keep the original backward walk, hop order and anchor
checks, including behavior on inconsistent intermediate coverage. Hop
distances come directly from the encoded dimensions instead of a strategy
lookup followed by a footprint lookup. The tradeoff is extra strategy
decoding at other `Get` sites and geometry encoding at `Set` sites. Source
inspection alone does not establish which tradeoff wins.

No owner, thread, cache, public format, floating-point operation, cost-table
layout, descriptor upload, grid-export validation or GPU kernel changes.
Both representations still allocate one search byte per base block and the
existing exported-grid owner. This is not a host-capacity or RSS reduction.
The diagnostic selector is not retained in production; no compatibility
layer is introduced.

The source audit reconstructs the original private grid after reversing the
explicit encoding and hop-decoding edits. It verifies exact source identity
outside that grid, apart from a recorded trailing blank line and diagnostic
export-name macros. Both DLL maps identify distinct original, candidate and
wrapper entry addresses. Per-query verification is absent from timed builds.

## Correctness gates

Release and scoped ASAN each pass:

- 2,590 exact output grids over 373 geometries and seven cost patterns;
- 21 matching extreme-cost errors with unchanged sentinel output;
- 23,499 invalid-input/failure-atomicity checks;
- eight concurrent independent search comparisons;
- 49,273,079 comparisons against the original boundary traversal.

Geometries include every width/height pair from 1 through 19 base blocks,
plus twelve thin, partial, HD and large cases. Costs cover zero, uniform,
random, exponent-range, discrete ties, neighboring floating-point values,
and maximum/minimum finite values. Unused candidate locations remain NaN.
Invalid visited DCT8 leaves, missing/extra table entries and null output are
checked without silently dropping the extreme-cost cases. The two compiler
runs agree on all reported counts. Totals are 5,180 exact grids, 42 matching
extreme errors, 46,998 failure checks, 16 concurrent checks and 98,546,158
boundary predicate comparisons.

The production-source search object is reused in both diagnostic DLLs;
only its prepared-cost export is renamed. Four labels share one DLL:
0/1 call the identical original entry, 2/3 the identical geometry entry.
A common wrapper measures the entire merge entry, including allocation and
validated export. Ordinary CPU-computed-cost search remains original.
The failed edge-bit candidate is not linked into either whole-encoder DLL.

Normal and scoped-ASAN preflights cover all eleven S169 inputs with automatic
and eight-thread budgets: 44 processes, 220 whole encodes including their
44 reference calls. The candidate and controls match exact frozen bytes
where available, a fresh original reference's complete summary, coefficient
width and native storage size on every call. The 22 ASAN processes execute
88 instrumented measured-DLL calls and 22 normal reference-DLL calls.
ASAN also instruments both search implementations in the host fixture;
it does not instrument every linked library or the CUDA driver.

Both whole DLLs contain the same eleven standard S169 GPU modules, verified
by extracted module hashes. This host-only pilot makes no new CUDA-sanitizer,
CTest, install-consumer, independent-decoder, Metal or Linux qualification
claim. Existing frozen predecessor evidence is not relabeled as new testing.

## Balanced whole-encode pilot

Four inputs run with automatic/eight-thread budgets, each twice with reversed
case and label-order schedules: 16 processes. Each checks one reference,
four warm rounds and twelve measured rounds of four labels. The Williams
orders are `0132`, `1203`, `2310`, `3021`, balancing positions and directed
immediate predecessors. This gives 1,040 whole encodes, 768 measured.
Combined with preflights, the stage checks 1,260 whole encodes.

The complete `Encode` call is timed; input I/O, backend construction,
reference generation, result clearing/checking and NVML queries are outside.
Results remain alive until the outer timer ends. All 41 existing phase
timers and the common merge-entry timer are retained. No samples are filtered,
and no run is restarted to obtain a favorable result.

A primary delta is the within-round mean of candidate labels 2/3 minus the
mean of original labels 0/1, followed by the median across twelve rounds.
Primary percentages use the corresponding within-round ratios. Each range
below spans four process cells, not a confidence interval. Negative is faster.
The machine-readable report also retains all four individual candidate versus
original pair medians, both duplicate comparisons, and every phase.

| Input | Merge delta (ms) | Merge change (%) | Complete-call change (%) | Faster primary cells: merge / whole |
| --- | ---: | ---: | ---: | ---: |
| Flower 500 | −0.016 to −0.001 | −4.46 to −0.28 | +0.03 to +2.02 | 4/4 / 0/4 |
| 1080p | −0.215 to +0.297 | −7.22 to +15.26 | −1.12 to +2.53 | 1/4 / 1/4 |
| 4K | −1.096 to +0.454 | −9.66 to +6.46 | −2.53 to +1.42 | 2/4 / 1/4 |
| Keong 3839 × 2159 | +0.240 to +1.188 | +2.14 to +8.53 | −3.87 to −1.20 | 0/4 / 4/4 |

Keong's complete-call gains accompany a slower merge in all four primary
cells; they do not establish a causal gain from geometry decoding. Flower's
small local savings accompany slower whole calls in every primary cell.
The 4K merge direction is mixed, and its individual candidate/control merge
deltas span −1.371 to +1.646 ms. The pilot does not pass the intended local
mechanism gate, so it is not expanded into a promotion sweep or a tuned
image-size/content cutoff. It establishes neither equivalence of performance
nor a universal slowdown. The original implementation remains enabled.

Quantization-minus-merge is calculated per encode before pairing. Its primary
4K changes span −8.759 to +5.506 ms, and Keong spans −5.853 to +2.327 ms.
These are useful variation bounds, not proof of a particular CPU/GPU cause;
independent medians are not subtracted to manufacture attribution.

All 2,048 warm/measured timing NVML endpoints report a 40,000 mW enforced
limit. That does not imply fixed clocks or thermals. No recorded build,
sanitizer, capture or source-archive/hash sweep overlaps the timing campaign.
Light editing and ordinary shared-machine activity remain limitations.

## Evidence and disposition

Root: `U:/gjxl-cuda-diagnostics/s174-artifacts`. Helpers and diagnostic sources:
`build-cuda-ninja/profiles/s174_*`. All output names are exclusive. The first
edge-bit source, each diagnostic/fixture revision, successful builds and four
rejected fixture jobs are preserved. All builds succeed; rejected jobs are
test failures, not blocked compilation or a privilege prompt. No predecessor
binary or pinned source is rebuilt or overwritten, and no material file is
deleted. The S173 inventory's 637 artifacts is reverified.

The accepted geometry fixture/build evidence, whole protocol/preflight,
raw timing logs, parsed measurements, paired report, source audit, decision,
source snapshots and final SHA-256 inventory are retained. There are 74
journaled jobs: 70 accepted and four rejected, including the later standalone
counterexample's successful build and reproduction. The initial unpinned source
audit needed exact trailing-newline and MSVC/clang map-format handling;
those checker corrections do not change the compiled candidate.

| Phase (UTC, September 10) | Start | Finish |
| --- | --- | --- |
| Initial host build | 04:32:21.056 | 04:32:43.598 |
| Geometry build and normal/ASAN fixtures | 04:38:58.896 | 04:39:27.733 |
| Whole build | 04:41:01.440 | 04:41:36.981 |
| Whole preflight | 04:42:38.546 | 04:43:37.515 |
| Unprofiled pilot | 04:45:29.164 | 04:48:25.193 |

Verification recomputes the report from raw sample logs, checks exact
schedule/counter/storage/power records, source/module/input/oracle hashes,
all journal outcomes and nonoverlap, and the frozen inventory:

```powershell
python build-cuda-ninja/profiles/s174_verify.py --frozen
```

No firewall/admin blocker was observed. No power, clock, thermal, priority,
affinity, driver, firewall or security setting changed. The encoder is not
established to be maxed out.

The next distinct hypothesis is device-side selection across independent
color tiles, consuming the already-resident packed costs and returning the
selected grid instead of reading/scattering every scalar cost for CPU merge.
This differs from S112's CPU scheduler and the two host cell encodings here.
It must preserve the original serial order within each tile, exact arithmetic,
strict tie rules, error precedence, failure atomicity, validated export and
metadata/ownership contracts. The extreme-cost counterexample is a warning
against assuming every intermediate grid is complete. No device selection
implementation or saving is claimed by this stage.
