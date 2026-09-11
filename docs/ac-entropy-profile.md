# AC entropy profiling against libjxl

Clustering is the largest remaining component of the balanced AC entropy task.
On identical captured populations with a 32-cluster cap, GJXL takes 1.40–1.73×
as long as libjxl. This is an implementation comparison, not a measured SIMD-only
speedup or a complete-encoder throughput comparison. A compact-storage experiment
preserving GJXL's arithmetic was slower and is rejected. No runtime change is
retained from this study.

## Baselines and measurement boundaries

Measured September 10, 2026 on Apple M4 Pro, 48 GiB, macOS 15.6, Release ARM64:

- GJXL `1d342c7`, `perf/entropy-bit-writing`, including the DC population reuse,
  exact clustering, word writer and producer-side ANS packing changes. The frozen
  baseline is `build/dc-population-reuse/candidate`.
- libjxl `a6afcb686ba32fa321904b6d5dbf1c688c95a69c`, the clean local
  `libjxl-gjxl-stage-profile` checkout and its `build/stage-profile-dev` library.
  This identifies the local comparison revision; it is not a claim about latest
  upstream. The retained build uses explicit Highway SIMD; its compiler flags
  disable automatic loop/SLP vectorization.
- Four canonical linear-sRGB float PFMs: Kodak17, Planter 1080p, Planter 4K and
  `padded-stress-4k`. The last is a corpus image, not the benchmark's synthetic
  `padded_4k` workload. Effort 7 and distance 1.2 in both encoders; fully-resident
  Metal and automatic CPU threads in GJXL, eight worker threads in libjxl.

These are identical input pixels and nominal settings, not matched decoded
quality. Coefficient populations, token counts, contexts and encoder policies
differ. In particular, padded-stress produces 5,312,893 GJXL AC tokens versus
16,618,143 libjxl AC tokens. Native stage durations cannot establish which encoder
is more efficient at equal quality or equal entropy work. CPU participant budgets
are also not identical.

Diagnostic copies instrument only the relevant translation units and link the
remaining baseline objects. Original builds and production sources are unchanged.
Three independent process pairs per input, alternating order, three warmups and
seven measured encodes follow one reference encode. Population capture happens
in the first reference encode, outside the retained samples. Native collection
contains 48 processes and 336 measured encodes. The buffering control adds 24
processes and 168 measured encodes.

Durations below are medians of process medians. Detailed timers are elapsed time
inside the individual AC task; other entropy tasks can run concurrently. They
must not be added to DC/order worker durations to infer entropy wall time.
Diagnostic logging and timers can perturb scheduling. Existing uninstrumented
workflow measurements provide the latency context, not a candidate speedup claim.

## GJXL attribution

All values are milliseconds. Population handling validates every one of 256 bins
per context, checks totals/extra bits/maximum symbol and creates borrowed views;
it does not tokenize coefficients again. Allocation and partition publication
each take under 0.01 ms in these measurements.

| Input | Population handling | Clustering | Coder selection | ANS model construction | HybridUint configuration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak17 | 0.264 | 0.944 | 0.074 | 0.372 | 0.013 |
| Planter 1080p | 0.775 | 4.733 | 0.210 | 0.593 | 0.020 |
| Planter 4K | 0.835 | 3.950 | 0.222 | 0.651 | 0.019 |
| Padded-stress 4K | 0.817 | 3.691 | 0.210 | 0.641 | 0.019 |

The prepared-value validation timer is another 0.002–0.005 ms; token-cost work
is zero in this population-based path. These selected subtimers do not exhaust
the AC task, which also performs model/context-map bookkeeping and publication.
The table does not attribute the remaining time to a particular function.

Uninstrumented workflow context:

| Input | Complete workflow | Serializer | Entropy wall | AC tokenization wall | Section writing wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak17 | 19.596 | 4.925 | 2.163 | 1.290 | 0.943 |
| Planter 1080p | 65.215 | 17.085 | 7.098 | 4.055 | 3.888 |
| Planter 4K | 198.735 | 33.980 | 6.331 | 9.768 | 8.331 |
| Padded-stress 4K | 195.777 | 30.584 | 6.130 | 9.400 | 7.556 |

AC finishes last among the entropy tasks in all 84 retained diagnostic encodes.
For Planter 4K, median DC completion is 3.206 ms after entropy-phase start and AC
completion is 6.505 ms. This supports prioritizing AC over another DC-only pass.

## Native libjxl attribution and streaming control

`BuildAndEncodeHistograms` includes initial population construction and
`BuildAndStoreEntropyCodes`. The latter includes clustering, uint configuration,
model construction/serialization and nested context-map coding. `Codes` therefore
includes `Clustering`; they are not additive columns. Initial population building
scans tokens, whereas GJXL consumes populations already collected during
tokenization. The GJXL population-handling column is not its direct counterpart.

Tiny nested metadata streams also use `LayerType::Ac`. Main coefficient timers
require that layer and more than 45 contexts. Their nested work remains included
in parent codes/total durations but is not counted as an extra main partition.
All four streaming partitions are summed per encode on each 4K input.

| Input | Main partitions | Final retained histograms | Initial populations ms | Clustering ms | Codes ms | Total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kodak17 | 1 | 21 | 0.366 | 0.496 | 0.938 | 1.394 |
| Planter 1080p | 1 | 32 | 2.604 | 2.691 | 3.566 | 6.417 |
| Planter 4K | 4 | 63 | 7.956 | 7.719 | 11.727 | 20.087 |
| Padded-stress 4K | 4 | 87 | 28.723 | 3.870 | 7.437 | 36.473 |

GJXL uses 2475 contexts for Kodak and 6930 for the other inputs, with 24/32/32/32
clusters. libjxl uses 1485 Kodak contexts and 6930 contexts per other partition.
Streaming libjxl can reuse previously encoded histograms using divergence costs
and a changing budget for new histograms. The final retained count above is not
the sum of per-partition histogram counts.

A separate harness sets `JXL_ENC_FRAME_SETTING_BUFFERING=0`. It produces one
main partition per encode, but also changes histogram-search work:

| Input | Final histograms | Initial populations ms | Clustering ms | Codes ms | Total ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak17 | 21 | 0.350 | 0.490 | 0.895 | 1.327 |
| Planter 1080p | 127 | 2.616 | 9.748 | 11.886 | 14.699 |
| Planter 4K | 128 | 7.470 | 9.081 | 11.254 | 19.018 |
| Padded-stress 4K | 128 | 28.993 | 7.821 | 9.611 | 38.872 |

This is not a GJXL slowdown or a libjxl regression. The experiment changes the
amount and type of model search. Neither this table nor the native table isolates
the cost of the two clustering implementations. HybridUint configuration is
under 0.004 ms across these libjxl cases and is not a material target here.

## Identical-population clustering replay

Replay loads captured histograms before timing and calls each implementation's
fast clustering routine with a 32-cluster cap and no prior output histograms.
Output allocation and initial Shannon-cost calculation are inside the timer;
input loading/conversion and result hashing are outside. GJXL's canonicalization
is included. The libjxl replay links the retained clustering object and Highway
library; an aborting stub satisfies its unused best-clustering cost dependency.
The fast path never calls that stub.

There are eight fixtures: one GJXL and one libjxl capture per input. For streaming
4K libjxl, the capture is its first main partition, not the whole frame. Both
implementations receive exactly the same counts within each row. Counts and
totals fit libjxl's int32 representation. Five alternating independent-process
pairs, five warmups and 31 measured calls give 120 processes and 3720 measured
calls including the compact experiment.

| Population source | GJXL ms | libjxl ms | GJXL/libjxl paired ratio | Compact GJXL time change |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 / GJXL | 0.777 | 0.556 | 1.40× | +14.04% |
| Kodak17 / libjxl | 0.739 | 0.466 | 1.57× | +1.17% |
| Planter 1080p / GJXL | 4.306 | 2.903 | 1.48× | +7.25% |
| Planter 1080p / libjxl | 3.910 | 2.684 | 1.47× | +6.56% |
| Planter 4K / GJXL | 3.637 | 2.113 | 1.73× | +4.75% |
| Planter 4K / libjxl first partition | 3.556 | 2.267 | 1.57× | +6.99% |
| Padded-stress 4K / GJXL | 3.402 | 2.256 | 1.51× | +7.02% |
| Padded-stress 4K / libjxl first partition | 2.408 | 1.494 | 1.61× | +11.35% |

Ratios and changes are medians of paired process-median ratios; dividing the
displayed medians need not reproduce them. Final cluster counts agree between
implementations on these fixtures, but assignments and arithmetic need not.
libjxl uses float Highway entropy/distance reductions and approximate logarithms;
GJXL uses ordered double arithmetic, exact-count logarithm lookup/fallback and
checked uint64 counts. The result isolates the overall clustering implementations
on the same input, not the individual benefit of SIMD, narrower counts or math.

The exact compact experiment repacks source populations into uint32 rows with
the partition's active alphabet width. It retains double cost order, ties,
cluster updates and canonicalization, and copies cached costs back to source
views. Packing allocation, copying and destruction are included in timing.
Large-count/small-context cases fall back to the original implementation.
All 40 GJXL/compact process pairs agree on hashes covering assignments, cluster
populations, metadata and raw double cost bits. Nonetheless every fixture has a
positive median paired time change. Reject the experiment; do not integrate its
storage ownership or run whole-encode qualification for a losing replay probe.
This result does not rule out compact populations produced directly at source.

## Decision and next work

The measured clustering gap merits further investigation, but copying libjxl's
float SIMD arithmetic would change the existing exact-output constraint. Even
matching its replay time on GJXL's Planter 4K populations represents about 1.5 ms
of isolated savings, roughly 0.8% of this complete workflow if fully realized on
the critical path. That is a scale estimate, not a measured throughput gain.

Continue with a bounded distance-kernel experiment on these saved fixtures,
preserving ordered double results if exact output remains required. Compare
instruction costs before adopting another layout or multi-distance rewrite;
the earlier paired-distance experiment and this compacting pass both failed to
beat the simpler retained code. A libjxl-style approximate SIMD policy would need
separate output-size and decoded-quality qualification, plus full-workflow and
batch confirmation.

Population validation is the smaller exact alternative: approximately 0.8 ms
on 4K. An internal producer-validated population interface could avoid redundant
checks, but public malformed-population rejection, all-bin validation and storage
contracts must remain covered. Do not simply stop checking bins beyond the
declared maximum. HybridUint configuration and another DC-only rewrite have much
less measured leverage. AC tokenization and section writing remain the larger
subsequent serializer boundaries.

## Evidence and validation

Artifacts are local and ignored under `build/ac-entropy-profile`: frozen source,
instrumented copies, compile/link commands, library/binary/input hashes, eight
population fixtures, raw timing files, process records and replay sources.
`native/summary.json`, `buffered/summary.json`, `replay/summary.json` and
`audit.json` retain the detailed results. `evidence-sha256.json` hashes diagnostic
sources, controllers and evidence files.

`python3 build/ac-entropy-profile/audit.py` verifies all 192 timed process records
against raw/log hashes and sample inventories, fixture count/extent validity,
replay medians and exact candidate hashes. All 12 native and 12 buffered libjxl
baseline/diagnostic output pairs match byte-for-byte. Four separate GJXL
baseline/diagnostic CLI output pairs also match. Harnesses check repeated output
against their reference encode. No cross-encoder byte identity is asserted.

The initial attribution pilot counted nested Ac metadata streams and is excluded
in `native-attribution-pilot` / `fixtures-attribution-pilot`. After correcting that
filter, the parser was extended to sum the four legitimate streaming partitions;
completed records were resumed without changing the measured binaries. Quiet
process guards passed before and after retained timed processes. They do not
establish exclusive-machine or controlled thermal conditions.

No production source changed, so the full Release/sanitizer qualification was
not rerun. This study validates diagnostic equivalence and rejects an isolated
experiment; it delivers no new encoder throughput improvement.
