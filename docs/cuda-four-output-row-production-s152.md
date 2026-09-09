# CUDA four-output-row production integration (S152)

## Scope and implementation

S151 qualified the S150 four-output-row bodies inside complete resident
encodes, but its uniform eligible-plane families did not exercise a mixed
full/half-resolution policy. S152 integrates that mixed policy into normal
source and qualifies the resulting production build. Starting commit:
`d8f1591`. Qualification passes, and the mixed four-row64/96 policy is
retained as a vertical-stage optimization. Complete-encoder throughput
remains inconclusive; this is not a claim of a dependable end-to-end gain.

The existing eligibility boundaries are unchanged: planes narrower than 32,
shorter than 96, or below 2,000,000 pixels use plain48. Eligible planes below
4,000,000 pixels use four-row64; larger eligible planes use four-row96.
The superseded three-row48/96 production bodies are removed, not retained
behind a compatibility switch. The internal forced-schedule test entry now
accepts 0/64/96 and rejects the old rolling48 value.

The new bodies preserve S150's 32-output-row chunks and 64-row shared ring.
Each warp computes the original ordered 33-tap normalization and broadcasts
it before any geometric predicate or block barrier. All twelve FMA chains
retain their original tap order. The owned S148 weight payload, plain
fallback, horizontal kernels, public API, compact-storage default, and
allocation/routing outside this vertical stage are unchanged.

The permanent low/medium test expands from 46 to 64 geometries. Added widths
33/65 cross row boundaries 127/128/129, 191/192/193, and 255/256/257, in packed
and padded layouts with five patterns and all four schedules (automatic,
plain, forced64, forced96). Forced schedules also explicitly check empty
dimensions and invalid input/output strides. Existing narrow/tall policy
fixtures, two-stream graph ownership tests, and independent reference
oracles are retained.

## Build and native identity

A fresh Windows Release build with CUDA 11.8 and MSVC 14.37 completes all
231 build steps. Qualification uses the RTX 3060 Laptop GPU (sm86); this
does not establish a performance range on other GPU architectures. Both production
rolling bodies exactly match the S150 prototypes: 2,352 static instructions,
47 registers, 24,576 shared bytes, zero stack/local allocation, and no local
loads or stores. The remaining 221 linked GPU bodies are unchanged. The
normal encoder still contains 223 bodies; tile128 is not introduced.

The four ordinary wide/compact release/host-ASAN encode executables have the
same ten GPU-module hashes as the clean production CLI. Release standalone
guard/scope programs link three of those modules. Their host-ASAN builds
link all ten because the build explicitly supplies instrumented pipeline
objects; those additional modules also match production exactly.

The diagnostic comparison starts from the new normal CUDA translation unit
and adds only the two exact old rolling bodies, a host-only family branch,
dispatch counters, and optional event brackets. The normal low/medium helper
retains its horizontal/vertical structure. The new family uses the actual
production selector; the old control independently implements S148's rule.
All 223 production bodies and both retained controls are native-exact in
both ordinary and event diagnostics. Every encode records its six actual
main/half call shapes and selected bodies.

Diagnostic selectors/counters are serial harness state, not production code
or a concurrency guarantee. The separate production oracle/batch tests use
normal executables without that state. Event builds add no synchronization;
the completed encode must make all twelve end events queryable.

## Preserved tooling corrections

The first CTest wrapper expected an older summary string. CTest itself
exited zero and reported all 84 tests passed, but the wrapper marked its
record unaccepted and the follow-on worker stopped before scoped tests.
The versioned validator checks all 84 individual `Passed` records, their
numbering/names, the actual summary, command, exit status, and original log
hash. No test is rerun to conceal the wrapper rejection.

The first diagnostic native audit assumed that both release and host-ASAN
standalone probes linked three modules. The ASAN probe actually links ten,
all hash-identical to the clean production encoder. The versioned audit
checks that exact full set for ASAN and the exact three-module subset for
release. Original rejected records, partial ELF extractions, and scripts
are preserved. Neither correction changes runtime source or system settings.

The first final verifier compared Windows-backslash snapshot paths against
forward-slash change names and therefore misclassified the intentional CUDA
edit as an unexpected change. `verify_s152_v2.py` normalizes that comparison
with `Path(...).as_posix()`. The failed original and the correction record
are preserved. No runtime source or measurement is changed or retried.

## Mixed-policy comparison protocol

The six frozen S151 cases and oracles are reused without resampling:
flower500, padded HD, flower2000, padded 4K, flower3200x2160, and
keong3839x2159. Both wide and compact AC storage are checked. This remains
the fully resident path at distance 1.2, effort 7, automatic CPU threads, and
final-score collection disabled, not exact-coefficient mode.

Four labels represent two duplicate labels per family: old0/2 and new1/3.
A randomized four-label Williams design uses four warmup and twelve measured
rounds, with reversed process order for the second repetition. Each timing
process checks 65 complete encodes, including its initial frozen-oracle
baseline; each preflight checks five. Twenty-four ordinary and twenty-four
separate event timing processes pass. Their measurements are never
pooled. Pairwise medians, all four cross-label differences, and duplicate
controls are retained, including the unchanged first horizontal interval.

The analyzer independently reconstructs policy selection, all six call
shapes, positional and within-round preceding-label balance, exact sample
counts, and ordered power/telemetry records. It validates completed preflight and memory phases
before timing begins. No clock, power, thermal, priority, affinity, firewall,
or other security setting is changed.

## Qualification result

All 84 CTests pass. Release and host-ASAN runs cover 5,120 guarded cases,
16 actual-policy fixtures, four tall fixtures, and 144 ownership graph
checks. Scoped host-ASAN covers another 120 fixtures and 720 comparisons.
Eight scoped CUDA sanitizer jobs cover both the scope and ownership probes
under memcheck, initcheck, synccheck, and racecheck; all are clean.

The normal production matrix checks 1,296 frozen-oracle encodes: 960 in the
ordinary qualification matrix, 222 under host-ASAN, 48 under CUDA memcheck,
and 66 under initcheck. This includes high-range compact-int32 fallback and
concurrent batches. Fourteen expected rejected inputs reproduce the prior
S148 errors; they are not counted as successful encodes.

The diagnostics check 3,370 more oracle encodes: 240 across 48 preflights,
ten across two event memory-check processes, and 3,120 across 48 timing
processes. All 20,280 event intervals, 3,380 telemetry records, 6,740 power
endpoints, and recorded shape/body/count checks validate. In total, S152
checks 4,666 complete encodes and eighteen CUDA sanitizer jobs (eight
scoped, eight normal production, two diagnostic).

There are 257 terminal job records: 240 accepted, fourteen expected input
rejections, and three preserved tooling rejections described above. The
230 GPU-facing processes are serial and do not overlap. Forty historical
runtime files and all frozen S151 files retain their hashes. The final
verifier independently recomputes the analysis and report; no unsuccessful
measurement is hidden by a retry.

## Performance result

The table reports the sum of the three full-resolution vertical intervals
per encode. Each range spans wide/compact storage and two opposite-order
process repetitions. Negative deltas mean faster; these are within-process
paired comparisons, not differences between the endpoints of the baseline
range.

| Case | Old baseline (ms) | New delta (ms) | New delta (%) |
| --- | ---: | ---: | ---: |
| Padded HD | 1.342-1.347 | -0.01485 to -0.00819 | -1.10 to -0.61 |
| Flower 2000 | 2.660-2.727 | -0.15027 to -0.10957 | -5.48 to -4.14 |
| Padded 4K | 7.333-7.727 | -0.76339 to -0.55834 | -9.86 to -7.14 |
| Flower 3200x2160 | 5.181-5.472 | -0.54656 to -0.22093 | -9.79 to -4.42 |
| Keong 3839x2159 | 7.266-7.379 | -0.66253 to -0.45978 | -8.82 to -6.17 |

All twenty eligible full-vertical primary comparisons and all eighty
cross-label comparisons favor the new policy. The sum of all six vertical
intervals also wins 20/20 primary and 80/80 cross-label comparisons.

Only the two 4K cases change their half-resolution bodies, from old48 to
new64. Padded 4K improves 1.03-7.71% (4/4 primary, 15/16 cross-label);
Keong improves 3.32-8.11% (4/4 primary, 16/16 cross-label). The other
half-resolution bodies remain plain and their timing changes are not
attributed to optimized work.

The combined horizontal-plus-vertical intervals favor the new policy in
18/20 eligible primaries and 66/80 cross-label comparisons. One HD and one
flower2000 primary are unfavorable. Both 4K cases win all eight primaries,
with 28/32 favorable cross-label comparisons. Their pair improvements span
2.18-6.00% for padded 4K and 1.95-6.21% for Keong.

### Complete encoder and controls

Ordinary whole-encoder timing is mixed: 12/24 favorable primary comparisons
and 48/96 favorable cross-label comparisons. The median primary percentage
is +0.072%, with individual primaries ranging from -5.62% to +4.89%.
Restricting to eligible cases gives 12/20 primaries and 48/80 cross-label
wins; the sixteen larger-image primaries win only nine times. All four
flower500 ordinary controls are unfavorable, with 0/16 cross-label wins,
despite executing identical GPU bodies. These observations are retained,
not dismissed as proven noise or assigned a cause such as RDP activity.

The largest absolute ordinary duplicate-label difference is 17.8654 ms
(new family, first wide flower3200 process). Separately, the event builds'
outer timings favor the new policy in 14/24 primaries and 44/96 cross-label
comparisons, with a largest absolute duplicate difference of 20.50725 ms.
Those instrumented outer results are not pooled with ordinary encodes.

Event controls also vary. Maximum absolute duplicate differences are
0.41677 ms for full vertical, 0.16128 ms for half vertical, 0.59341 ms for
horizontal, and 1.11565 ms for the combined pairs. The unchanged first
horizontal interval, before any changed vertical work, has a 0.03891 ms
maximum absolute duplicate difference. Its Keong primary percentages range
from -3.05% to +4.12%. Unchanged horizontal work is favorable in only 8/20
eligible primaries, with an unfavorable primary reaching +5.53%. These
controls are not used as exact normalization factors.

Across both 4K cases, the changed six vertical intervals save about
0.50-0.89 ms per instrumented encode. Their old interval sum is only
2.90-3.60% of the instrumented outer baseline, and the savings amount to
roughly 0.16-0.31% of that baseline. This is descriptive scale, not an
estimate of ordinary whole-encoder speedup.

All 6,740 recorded power-limit endpoints are 40,000 mW, and all six
telemetry query statuses succeed on every record. Reported SM endpoints
span 210-1,770 MHz. Of 3,380 records, 3,379 report pstate 3, memory 5,500 MHz,
and throttle mask 0x24; one reports pstate 5, memory 810 MHz, and mask 0x1.
No observations are discarded and no system settings are changed. Endpoint
telemetry does not establish the clock during a particular interval or
identify a unique cause of timing variation. No admin/firewall blocker
appears during this campaign.

## Retention decision and next work

Retain mixed four-row64/96 as a production vertical-stage optimization:
the exact S150 bodies preserve output contracts and show consistent stage
improvements against actual old dispatch inside complete encodes. Remove
the superseded rolling implementation without a compatibility layer.
Do not claim a dependable whole-encoder throughput gain.

The existing 2M/4M boundaries remain a conservative policy, not a newly
optimal crossover. S151's flower2000 preference for tile64 is still an
unresolved tuning opportunity; this integration does not retune it.
Tile128 is not retained. Compact coefficient storage remains opt-in and
unrelated allocation/routing is unchanged.

Refresh the fully resident critical-path profile before investing further
in sub-millisecond convolution tuning. Earlier Malta, erosion/reduction,
and fused AC stages remain candidates, but historical stage times are not
a fresh bottleneck ranking after the retained changes. The backend is not
claimed to be maxed out.

## Evidence and reproduction

Local evidence is under `build-cuda-ninja/profiles/s152-artifacts/`.
`before.json`, `production_inputs.json`, and `protocol.json` pin the source,
build/input identities, and ordered campaign. `native.json` and
`diagnostic_native_v2.json` record exact body/module checks. The production
matrix reports, `preflight.json`, `memory.json`, `screen.json`,
`analysis.json`, and `report.json` retain qualification and all comparisons.
Rejected original scripts/logs and `verifier_correction.json` remain in
the evidence closure.

The final source snapshot and artifact inventory are produced only after
qualification, analysis, and documentation are complete. Revalidate the
frozen evidence with:

```powershell
python -X utf8 build-cuda-ninja/profiles/verify_s152_v2.py --frozen
```

`freeze_s152.py` creates the exclusive snapshot/index, final summary, and
SHA-256 inventory. It is a one-time finalizer, not a command to rerun over
an existing frozen campaign. Historical builds and evidence are untouched.
