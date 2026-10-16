# GPU histogram fitting: reuse across changing buffer sizes

This extends the [submission experiment](gpu-histogram-submission-experiments.md)
with bounded reuse of larger output, dispatch and scratch buffers. The new
private control is `--fit-buffers=capacity`, now the default whenever GPU fitting
is selected. Existing `fresh` and exact-size `reuse` controls remain available;
public routing is unchanged. All comparisons use one compute encoder.

## Results

Capacity reuse removes most repeated buffer allocations and their associated
reported driver scheduling interval. All correctness and resource checks passed.
The fixed-work probe supports the mechanism; complete-image improvements vary
with the image and runtime conditions. These results support capacity reuse as
the GPU fitter's default buffer policy. Selecting GPU fitting remains a separate
choice: this study compares buffer policies on the same private GPU route and
does not establish a speedup over CPU fitting or libjxl.

The matched battery study ran on an Apple M4 Pro, 14 CPU cores, 48 GiB RAM,
macOS 27.0.1. Battery `powermode` was 1 at the initial, intermediate and final
settings checks; battery charge went from 90% to 79%. Power-source checks bracket
each capture. The earlier AC study is a separate comparison, not the timing
baseline for this change.

### Fixed work with changing sizes

These are medians of eighteen measured five-evaluation sequences, in
milliseconds. Each sequence includes its first cold allocation.

| Buffer policy | Evaluation total | Observed range | Reported driver scheduling total |
|---|---:|---:|---:|
| Fresh | 96.289 | 91.762–108.115 | 5.559 |
| Exact-size reuse | 94.747 | 92.969–116.796 | 5.646 |
| Capacity reuse | **90.081** | 87.531–114.560 | **1.913** |

Capacity reuse reduced the median evaluation total by **6.4% versus fresh**
and **4.9% versus exact-size reuse**. Its reported driver scheduling total fell
65.6% versus fresh. All three buffers were reused for steps two through five;
the deliberately alternating sizes produced no exact-size hits. GPU execution
totals were similar: 86.285, 84.741 and 85.305 ms respectively. The timing ranges
overlap; the directly observed allocation and scheduling reductions are stronger
evidence than a precise throughput prediction.

### Complete-image timing

Values are medians of six measured encodes, with minimum–maximum in parentheses,
in seconds. The supplemental 1049 cohort is kept separate from the initial
comparison.

| Image / cohort | Fresh | Exact-size reuse | Capacity reuse |
|---|---:|---:|---:|
| 1029, initial | 6.221 (5.455–6.546) | 5.789 (5.141–6.336) | **5.247 (5.097–6.024)** |
| 1039, initial | 3.972 (3.581–5.185) | 3.633 (3.493–4.118) | **3.589 (3.331–3.954)** |
| 1049, initial | 12.250 (4.526–14.312) | 12.647 (4.932–14.646) | 12.153 (6.598–13.722) |
| 1049, confirmation | 9.966 (9.072–12.732) | **9.554 (9.135–12.211)** | 9.574 (8.803–10.095) |

On 1029 and 1039, capacity reuse was faster than fresh in all six paired rounds.
Its medians were **15.7% and 9.6% lower than fresh**, and **9.4% and 1.2% lower
than exact-size reuse**, respectively.

The initial 1049 cohort shifted between roughly 4–7 s and 12–14 s regimes.
All observations are retained; its capacity result beat fresh in only three of
six paired rounds. The separately announced confirmation produced a 3.9% lower
capacity median than fresh, with four of six paired rounds faster, but essentially
the same median as exact-size reuse. It does not demonstrate an additional
end-to-end benefit from larger reuse on 1049. The original three-image sums of
medians are 22.443 / 22.069 / 20.989 s; the instability makes that aggregate an
unreliable summary of the expected gain. The confirmation is not substituted
into that aggregate.

### Complete-image attribution

The separate traced cohort contains two measured encodes per image and policy.
The following sums cover those six encodes per policy; warmups are excluded.

| Metric | Fresh | Exact-size reuse | Capacity reuse |
|---|---:|---:|---:|
| GPU batches | 3,299 | 3,317 | 4,319 |
| Output / dispatch / scratch reuse | 0 / 0 / 0% | 33.01 / 0.87 / 14.86% | **99.63 / 99.56 / 99.77%** |
| Reported driver scheduling, total | 6.877 s | 6.366 s | **0.826 s** |
| Reported driver scheduling, mean per batch | 2.085 ms | 1.919 ms | **0.191 ms** |
| Host allocation phase, total | 0.607 s | 0.741 s | **0.083 s** |
| Cumulative newly allocated backing | 55.921 GiB | 45.076 GiB | **0.658 GiB** |
| Idle backing after each encode, range | 0 MiB | 4.79–7.53 MiB | 52.24–62.92 MiB |

The driver interval fell **90.8% per batch** and **88.0% in total**, despite
30.9% more batches than fresh. Logical GPU nodes, rows and properties, CPU-node
work and encoded bytes were unchanged. Faster evaluation changes which CPU
fits are ready together, producing more, smaller batches. These full-workflow
totals therefore include scheduling feedback; the fixed-work probe isolates
the buffer policy more directly.

The driver interval is `kernelEndTime - kernelStartTime`, an elapsed CPU-driver
scheduling interval, not GPU shader time or measured CPU busy time. It can
overlap GPU execution. It is not an exclusive additive component of encode wall
time, and it does not identify a particular VM or driver-internal cause.
Cumulative new allocation sizes likewise measure neither peak RAM nor memory
traffic. The tradeoff is retaining approximately 52–63 MiB of idle backing;
all accepted encodes passed the post-timing trim and zero-resource checks.

Across the image phases, **111 accepted encodes** passed independent decoding
and output-hash checks, including 21 excluded warmups. Every output matched the
other policies and the prior committed submission study byte for byte. There
were no capacity-admission retries or GPU-admission fallbacks. The variable-size
probe passed **315 exact evaluations**, including 45 excluded warmups.

Two overlapping attribution attempts were rejected and retained. Accepted runs
had no detected competing codec/build/test/capture work, but desktop and browser
activity continued. Waiting for another experiment also separated parts of the
study. Power matching and balanced order did not prevent substantial runtime
drift, whose cause was not established.

## Storage and execution contract

The three histogram cache slots accept backing whose capacity is at least the
logical request size. A cache miss allocates exactly the requested size. This
does not round allocations up, introduce size classes, or preallocate a global
maximum. When concurrent owners return different sizes to the same slot,
capacity mode prefers the larger backing within the existing shared cache
limit. Other operations retain their existing exact-size acquisition and
retention policy.

Larger backing requires larger admission. Before reserving each evaluation,
the coordinator obtains an advisory snapshot of the three cache capacities and
adds any excess to the ordinary evaluation storage bound. The backend can use
only backing within those per-buffer limits. A concurrent return of a still
larger allocation therefore cannot silently enlarge the admitted envelope.
The snapshot does not pin or activate cached storage.

If the expanded reservation is unavailable, the coordinator retries the
original logical bound, with larger reuse disabled for that batch. CPU fallback
remains the existing response if evaluation admission itself is unavailable.
Per-buffer activation also rejects a cross-domain allocation, reclaimed backing,
or a transfer that exceeds the current reservation, then attempts an exact-sized
fresh allocation. Reservations never grow during execution.

Logical request lengths, offsets, dispatch grids, scratch layout, property
results and kernel arithmetic are unchanged for a given batch. Faster evaluation
can change when independent CPU fits become ready, so the complete workflow can
form different batch shapes and submission counts despite identical logical
node/row/property work and final bytes. The full backing capacity remains
charged while live or idle; only requested-byte accounting shrinks for a smaller
request. Scratch reuse also respects the current scratch ceiling. Result backing
cannot reenter the cache while its result owner exists. Failure draining,
purgeability, cache generation checks, trimming and eviction remain in force.

The pure `ComputeHistogramFittingCapacityBound` helper checks undersized limits,
unused-buffer limits and overflow without modifying its output on failure.
Zero capacity limits mean exact logical sizes. Direct callers that want larger
reuse must obtain and admit the capacity snapshot, then pass it to evaluation.

Schema 11 adds logical requested bytes and actual capacities for each role,
plus the number of coordinator retries with the original reservation. Existing
reuse bits identify actual reused backing. `reused_buffer_bytes` continues to
count logical requested bytes served by reuse; actual capacity is recorded
separately. Summing these capacities across batches measures cumulative use,
not resident memory.

## Qualification method

The study uses the same 4000 × 3000 RGB8 photographs 1029, 1039 and 1049,
learned Best policy, Metal unique sample rows, a 16,384-row GPU cutoff,
64-fit cohorts, four histogram shards, 64 active properties and a 64 MiB
logical scratch ceiling. CPU limits remain 14, with four participants for
CPU weighted features and an 8 GiB execution domain. Backend/domain creation,
input/output I/O, independent decoding, hashing and final cache trimming are
outside the complete private encode interval.

Every image configuration receives one excluded warmup. The uninstrumented
comparison uses six measured repetitions per configuration. The three policies
follow a six-round order that balances positions and ordered adjacent pairs.
The separate traced study uses two measured repetitions per configuration;
its timing is attribution evidence, and does not estimate instrumentation
overhead by subtraction from the uninstrumented cohort.

After observing the initial 1049 runtime shift, a supplemental uninstrumented
1049 cohort used one excluded warmup and six measured repetitions per policy,
with the same balanced order. The phases contain 63 initial comparison encodes,
27 attribution encodes and 21 confirmation encodes, respectively. No initial
observation was removed or replaced by the supplemental cohort.

The synthetic probe runs the same sequence under each policy:
32 nodes × 131,072 rows, 1 × 16,384, 10 × 32,768, 1 × 16,384,
and 32 × 131,072. Each policy starts each sequence with an empty cache.
Immutable inputs, memberships and parent summaries are held fixed. There are
three excluded warmup rounds and eighteen measured rounds, with balanced policy
order. Evaluation timing excludes preparation, admission, the native CPU oracle
and result destruction; result destruction is recorded separately. The first
cold evaluation remains included in the sequence total.

The runners record power, thermal state, leading processes, binary/dependency/
source hashes, exact output hashes and competing jobs. Detected overlapping
codec/build/capture work causes an attempt to be rejected and retained. This
does not imply absence of unrelated system background activity.

## Correctness and resource checks

Six focused Release suites passed: operation planning, fit queue, Metal
evaluation, complete-tree workflow, resource budgets and cache admission.
The expanded evaluator reports **401 exact batches, 7,937 property results,
19 complete trees and 31 failure checks**. It covers both encoder
layouts for fresh, exact-size and capacity reuse.

New tests exercise large→small→large requests, unchanged logical result lengths,
actual capacity charging, a held larger output, a smaller late release,
stale capacity snapshots, tight-budget rejection followed by exact allocation,
cross-domain cache rejection, overflow, and complete charge cleanup. Existing
tests also exercise concurrent evaluation, trimming generations, reclaimed
scratch and submission/completion failures. The workflow test checks repeated
capacity-enabled fitting against native CPU trees.

No new sanitizer or Metal shader-validation run is claimed.

The subsequent default promotion changes the operation options and profiler
default to capacity reuse. GPU profiler output records the effective policy even
when `--fit-buffers` is omitted. The existing variable-capacity and workflow
tests now exercise default-constructed options; the fixed-size probe explicitly
retains its fresh/exact-size controls. The reported measurements were captured
before promotion with explicit policy flags and frozen binaries, so their
provenance and results remain unchanged.

After promotion, all six focused Release suites passed again. Four full-resolution
1029 encodes (omitted flag, explicit capacity, exact-size reuse and fresh) decoded
exactly and matched the prior qualified bytes. The omitted flag reported capacity
reuse and actually reused larger backing; exact-size and fresh overrides retained
their original behavior. The fixed-size probe passed 48 exact evaluations across
fresh/reuse and single/multiple-encoder controls. These are correctness checks,
not a new performance comparison; commands, output hashes and logs are retained
under `build/gpu-histogram-capacity-default-20261016/`.

## Reproduction and artifacts

Worktree: `gjxl-parallel-tree-fitting`, branch `feat/gpu-histogram-fitting`.
Base revision: `5053b45`, plus the recorded working diff. Artifacts are under
`build/gpu-histogram-capacity-reuse-20261016/` and include frozen binaries,
source/dependency hashes, build/test logs, per-run outputs and independent
decode results. `analyze.py` audits hashes and generates the
[machine-readable summary](gpu-histogram-capacity-reuse-results.json).
`qualification/results.json`, `attribution/results.json` and
`confirmation/results.json` retain all image observations and rejected attempts;
`variable-batches/manifest.json` records the probe. Initial settings are in
`provenance.json`; `final-environment.json` records the closing power/settings
and thermal snapshot.

```sh
python3 tools/modular_reference_learning/run_histogram_submission_probe.py \
  --binary build/gpu-histogram-capacity-reuse-20261016/probe \
  --out NEW_PROBE_DIRECTORY --mode=capacity --repeats=18 --power=battery
```

The complete-image runner is `run_gpu_histogram_wait_study.py`; the retained
`qualification-variants.json` and `attribution-variants.json` contain exact
commands. Qualification uses `--gpu-profile=gpu-off`. Attribution uses
`--gpu-profile=gpu-wall --gpu-wait-trace=on`.
