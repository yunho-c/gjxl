# Direct packed strategy-cost consumption (S135)

**Disposition: not promoted as a standalone speed optimization.** A packed
cost consumer removes dense host storage and scattering, and lowers the
enclosing strategy-search time in all sixteen primary comparisons. Complete
encoding improves in eight and regresses in eight, with substantial duplicate
control variation. The tested candidate is archived; runtime source is
restored to S134 (`ec9a9b5`). No compatibility helper or experiment selector
is retained in production. The encoder is not established to be maxed out.

## Hypothesis and implementation

S133 identified host candidate construction and cost merging as GPU-idle
intervals. The GPU evaluator already returns contiguous scalar costs ordered
by strategy, color tile, and local anchor. The frontend nevertheless fills
seven image-wide arrays with NaNs and scatters those packed results to global
base-block positions. The hierarchical CPU search then visits one color tile
at a time. That extra representation is not required for the search's math.

The candidate passes each packed result span directly to the existing
internal candidate-cost entry. It validates exact packed counts from the
fixed policy's footprints and anchor steps, including partial and thin tiles.
Before searching a tile it computes seven local pointers, row strides and
anchor shifts. Leaf lookup uses those tile-local views. Search arithmetic,
merge order, priorities, strict comparisons, tie-breaking and final grid
export remain unchanged. A private grid still commits only on success.

The candidate removes seven dense vector owners, their initial NaN fill and
the packed-to-dense scatter. Candidate descriptors, matrices, device arenas,
transfers, kernels, compact policy and CPU thread scheduling are unchanged.
It does not cache data across independent encodes. The internal table format
is replaced directly; no dense-format compatibility route is added to the
candidate runtime.

| Input / padded blocks | Duplicate dense bytes removed | Packed cost bytes retained | Descriptor bytes still constructed/uploaded |
| --- | ---: | ---: | ---: |
| Flower 500 / 63 × 63 | 111,132 | 63,516 | 381,096 |
| HD / 240 × 135 | 907,200 | 521,520 | 3,129,120 |
| 4K / 480 × 270 | 3,628,800 | 2,088,480 | 12,530,880 |
| Flower 2000 / 250 × 250 | 1,750,000 | 1,003,692 | 6,022,152 |

These are source-derived storage/transfer byte counts, not RSS, pool reserve
or measured transfer latency. The within-process diagnostic verifies the
dense owner's exact live capacity for each label. The savings are not enabled
in the restored production checkout.

## Correctness and isolation

Starting revision: `ec9a9b5`, branch `feat/cuda`, September 8, 2026.
Windows, RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37, Release; scoped
host ASAN uses clang-cl. The experiment reuses frozen S134 libraries and
S132's unchanged compact/ASAN resident-owner objects. It compiles the actual
candidate codec and GPU search translation units and links separate
production-source diagnostic encoders. It is not a new full CMake build.

A renamed copy of the original codec implementation provides the dense-table
oracle. Its four exported function names are changed at compile time to avoid
symbol collisions; its search implementation is unchanged. The table struct's
layout is identical in both versions, although the candidate changes its
internal ordering contract. Anonymous implementation types are confined to
their respective translation units.

Release and scoped host-ASAN each pass 1,492 exact-grid comparisons and
14,920 unchanged-output rejection checks, plus sixteen concurrent independent
comparisons. Geometries include every width/height pair from one through
nineteen base blocks, twelve additional thin/edge/HD/4K cases, and four
concurrent geometries. Zero, uniform, random and exponent-range costs cover
ties and varied decisions. Packed fixtures use an independent copy of the
previous producer's nested enumeration, not the new count helpers. Missing
or extra span elements and nonfinite/negative visited DCT8 leaves test failure
atomicity and matching errors. The diagnostic exponent-range generator has
two RNG calls in one expression; do not treat its aggregate grid hash as a
portable cross-compiler oracle without first fixing evaluation order.

The whole-encode diagnostic retains one common GPU search owner with an
additional dense vector array used only by baseline labels. Candidate labels
leave it empty. Conditional setup/scatter dispatch selects the appropriate
codec consumer, whose baseline/candidate linked addresses are distinct.
The unused member is shared diagnostic overhead, not part of the candidate
production structure. There are no conflicting definitions of the opaque
prepared owner in different translation units. Timers and selector counters
are diagnostic-only; they are not committed to runtime code.

Sixteen whole-encode preflights and sixteen timed processes check 1,376 frozen
codestreams, forty under scoped ASAN. Sixteen additional production-source
encoder preflights check 240 frozen codestreams, 120 under scoped ASAN. These
cover small/4K quality batches, rate search/prepared reuse, and two-request
4K batches in both coefficient widths. Total: **1,616 frozen-oracle encodes,
including 160 scoped host-ASAN encodes**. ASAN covers the modified codec/GPU
search, caller, resident owner and S134 pipeline units, not every library.
All eight linked GPU executables use the same ten S134 modules: all 214
current GPU bodies are unchanged. There is no new independent decoder,
CTest, CUDA sanitizer, Metal or Linux qualification claim for this
unpromoted experiment.

## Balanced complete-encode timing

Inputs are the same as S133/S134: Flower 500, HD from 1919 × 1079, 4K from
3839 × 2159, and Flower 2000. The latter is fourfold nearest-neighbor
replication of the 500-square crop, not a native 2000-square photograph.
Distance 1.2, effort 7 and fully resident encoding are fixed. Wide and
opt-in compact modes are separate processes; no compact default changes.

Labels 0/2 are duplicate baselines, 1/3 duplicate candidates. Randomized
four-label Williams sequences balance positions and immediate predecessors.
Each process checks one reference, four warmup rounds and sixteen measured
rounds: 81 encodes, 64 measured. Four inputs, two widths and two reversed
repeats give sixteen timed processes and 1,024 measured encodes. Byte/summary
equality, coefficient width/storage, branch counts and host capacity are
checked on every encode. Input I/O, backend construction and result checks
are outside the complete-call boundary; returned results remain alive until
after its timer stops.

The table reports repeat 0 / repeat 1. A delta is the median across rounds
of the mean of candidate labels minus the mean of baseline labels. Negative
means faster; these pairs are not confidence intervals or differences of
aggregate medians. Strategy-search time includes its host preparation,
GPU submission/wait, cost readback and CPU merge. Scatter time isolates the
host scatter/assignment after each readback.

| Input / width | Complete-call delta ms | Strategy-search delta ms | Scatter delta ms |
| --- | ---: | ---: | ---: |
| Flower 500 / wide | +0.189 / −0.531 | −0.018 / −0.091 | −0.033 / −0.037 |
| Flower 500 / compact | +0.181 / +0.159 | −0.019 / −0.009 | −0.035 / −0.035 |
| HD / wide | −0.959 / −0.498 | −0.596 / −0.675 | −0.346 / −0.326 |
| HD / compact | −1.558 / +0.305 | −0.742 / −0.588 | −0.349 / −0.337 |
| 4K / wide | −2.578 / +0.513 | −4.740 / −1.293 | −1.431 / −1.518 |
| 4K / compact | +2.907 / −4.107 | −1.317 / −3.286 | −1.419 / −1.531 |
| Flower 2000 / wide | −1.507 / +3.550 | −1.352 / −1.674 | −0.735 / −0.690 |
| Flower 2000 / compact | −1.794 / +1.290 | −0.987 / −1.708 | −0.658 / −0.662 |

Strategy-search direction is favorable in 16/16 primary and 60/64
candidate/baseline label-pair medians; scatter is favorable in 64/64 pairs.
Merge alone improves in 10/16 primary and 44/64 pair medians. The complete
call is favorable in only 8/16 primary and 36/64 pair medians; enclosing
quantization is favorable in 9/16 primary and 38/64 pairs.

Whole-call duplicates are substantial: at 4K, baseline duplicate medians
range from −6.71 to +4.24 ms, candidate duplicates from −13.47 to +12.63 ms.
The small-image candidate's median regressions also cannot simply be
discarded. Later phases change by more than the isolated scatter saving in
several runs. These results neither establish a dependable encoder speedup
nor prove that the packed representation caused the slower whole calls.
Do not subtract independent medians to manufacture an attribution.

All 2,752 within-process power-limit endpoints are 40 W. This does not prove
constant clocks or absence of transient changes. Measured jobs overlap no
other recorded job; lightweight read-only analysis/tool activity remains a
shared-machine limitation. No firewall/elevation blocker appeared, and no
power, clock, priority, affinity, firewall or privilege setting was changed.

## Evidence and next action

Root: `U:/gjxl-cuda-diagnostics/s135`; drivers: `build-cuda-ninja/profiles/s135_*`.
The exact unpromoted runtime is under `candidate/`, indexed by
`candidate_index.json`; `before/` records S134. Main evidence is
`within_analysis.json`, `timing_summary.json`, `encode_preflight.json`,
`linked.json`, `descriptor_plan.json`, source snapshots and the SHA-256
manifest. All 55 recorded jobs complete successfully. The runtime restoration
is checked against the original source and Git revision. Forty retained
runtime hashes and historical input/oracle/library hashes are unchanged.
No failed job was silently restarted and no artifact was overwritten.

Reproduction needs a new artifact root and the archived candidate sources;
the current production checkout is deliberately not this candidate. Preserve
the original/candidate distinction when regenerating the A/B translation
units. The tested cost-layout prototype remains available for a larger
representation experiment, but this study does not justify promoting it
solely on its local timer or 3.46 MiB 4K dense-storage saving.

Next target candidate descriptor construction and upload. At 4K the source
enumerates 522,120 descriptors, each 24 bytes. With resident quantization and
CfL, only regular coordinates and policy constants need to be represented;
the device supplies the image-dependent fields. Investigate generating those
descriptors in the existing quant-norm preparation pass, or deriving them
implicitly in consumers. The former retains device descriptor storage but
could remove host construction/upload; the latter removes storage too but
adds indexing work to several transform consumers. Both require exact edge
ordering, range/alias validation, unchanged arithmetic and whole-encode
tests. These are source-grounded hypotheses, not implemented savings.
S112's rejected parallel scheduler remains unpromoted.
