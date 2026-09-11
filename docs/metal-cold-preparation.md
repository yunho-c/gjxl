# Cold Metal reference preparation

The retained change reduces requested device storage by **63.2 MiB at 4K** by
sharing two AQ residual-coefficient plane capacities with Butteraugli. Its cold
latency benefit is **not yet established**: the primary untraced campaign shows
median paired improvements of 3.9–4.5 ms, but a separate traced campaign does not
reproduce a consistent gain. Warm encoding is approximately unchanged.

Base: `34135b269b5091b7314d50cdbe389d0cf447aaa9`, including the qualified AC
scratch optimization. Branch: `perf/metal-cold-preparation`. Measurements used
an Apple M4 Pro on September 11, 2026. Frozen source, binaries, scripts, raw
records, and rejected-prototype evidence are under `build/cold-preparation/`.

## Attribution of the cold driver interval

Reference construction borrows image and convolution scratch from AQ. Its first
submission therefore uses both the approximately 830 MiB Butteraugli arena and
642 MiB AQ staging arena. Space reserved in these backing allocations for later
stages also contributes to first GPU use.

A diagnostic uses one selected arena in an earlier submission: initialize one
word before normal population, then run a one-element affine identity through
an existing shader. The first-use driver cost moves to that submission.

| Arena used earlier | Reference driver interval | Earlier submission driver interval |
| --- | ---: | ---: |
| None | 45.524 ms | — |
| AQ persistent | 43.074 ms | 7.813 ms |
| AQ staging | 21.298 ms | 21.661 ms |
| Butteraugli | 16.201 ms | 28.028 ms |

These are three-process medians on padded 4K with preparation caches trimmed
before every encode. The columns are independently summarized and are not
additive. This establishes a first-use cost per backing buffer; it does not
identify individual driver page-wiring functions. No new Instruments stack
capture was performed. Earlier binding is an attribution probe, not a retained
optimization or a demonstrated complete-call saving.

## Retained storage change and lifetimes

Butteraugli borrows **11 disjoint planes instead of nine**. The two additional
views replace distorted-image band slots 10 and 11. The 33 logical slots,
shader arithmetic, cached reference features, and reference masks are preserved.

| AQ storage | Lifetime around a comparison | Treatment |
| --- | --- | --- |
| Two filter images, six planes | Filtering completes before comparison | Existing borrowing |
| Gathered pixels, three plane capacities | Forward transform completes before comparison | Existing borrowing |
| Residual coefficients, last two plane capacities | All inverse transforms finish before comparison; next reconstruction writes them again | Two additional borrowed views |
| Residual coefficients, first plane capacity | Complete-distance-map diagnostics can write it during comparison | Kept separate from borrowed views |
| Reconstructed XYB, three planes | Deferred frontend reconfiguration can read it again to rebuild forward coefficients | Kept separate from borrowed views |

The mapping is centralized in `metal_storage_plan.h` and used by allocation and
binding. Existing validation checks backend ownership, capacity, pairwise
disjointness, and reference non-overlap. AQ retains ownership until its prepared
Butteraugli borrower is destroyed. Small images and filter configurations that
cannot lend the planes retain independent storage. There is no new allocation
cache, dispatch, synchronization boundary, or shader change.

| Requested device allocation | Before | After |
| --- | ---: | ---: |
| Butteraugli, 3839×2159 | 870,620,980 B / 830.289 MiB | 804,313,652 B / 767.053 MiB |
| Butteraugli, 3840×2160 | 871,171,636 B / 830.814 MiB | 804,816,436 B / 767.533 MiB |
| AQ staging, either input | 672,898,048 B / 641.726 MiB | Same |

The reductions are **66,307,328 B / 63.236 MiB** for padded 4K and
**66,355,200 B / 63.281 MiB** for exact 4K, including alignment. Input and AQ
persistent allocations are unchanged. These are allocation capacities, not RSS
measurements. The storage is removed from the complete workflow, rather than
allocated at a later stage.

### Rejected larger prototype

An initial five-plane prototype additionally borrowed the three reconstructed
XYB planes. It saved approximately 158 MiB and measured 8.9–10.1 ms faster cold
encodes. Although 38 corpus comparisons and focused tests passed, the existing
`metal_completed_frame` test exposed invalid device numerics after deferred
frontend reconfiguration. `Reconfigure` invalidates forward-coefficient state;
regenerating it can read `reconstructed_` again after Butteraugli has run.

That prototype was rejected. Its storage and speed numbers do **not** describe
the retained change. The existing test was preserved, and the revised two-plane
change passes it. The rejected patch, report, timings, and failure logs remain
as evidence of the lifetime constraint.

## Complete encoding measurements

Release, fully resident Metal, SIMD/fused-tuned implementation, effort 7,
distance 1.2, automatic CPU threads, no requested final score. Six independent
process rounds alternate arm order and shuffle workload/cache cases. Each
process performs an initial validation encode, three warmups, and five retained
samples: 48 processes and 240 retained complete-call samples. Tracing and GPU
profiling are disabled for this primary campaign.

**Cold allocations** means the existing cache-trim API runs before each encode,
outside the timed boundary. Code and GPU pipelines are warmed; this is not full
process startup. **Warm** means sequential reuse without trimming. The public
encoding boundary includes frontend, AQ, frame handling, and CPU codestream
work. Input loading, filesystem output, backend construction, and trimming are
excluded. All measurements and process snapshots are retained, including
outliers. Normal desktop activity continued; no other study build, test, or
benchmark ran concurrently with the measurement campaign.

| Workload / cache | Baseline median | Candidate median | Median paired delta | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Padded 4K / cold allocations | 251.726 ms | 248.123 ms | −4.489 ms / −1.76% | 5/6 |
| Planter 4K / cold allocations | 258.647 ms | 253.665 ms | −3.891 ms / −1.53% | 4/6 |
| Padded 4K / warm | 183.653 ms | 183.344 ms | +0.170 ms / +0.09% | 3/6 |
| Planter 4K / warm | 186.215 ms | 186.841 ms | +0.625 ms / +0.34% | 1/6 |

Values are medians of process medians. Paired differences and ratios are
computed within each round, so columns need not subtract. Cold paired ranges
are −26.065 to +46.697 ms for padded 4K and −48.355 to +4.324 ms for planter.
Warm ranges are −2.175 to +7.805 ms and −7.233 to +1.937 ms respectively. The
specific causes of these untraced outliers were not measured.

## Separate diagnostic campaign

Three rounds per arm, input, and cache protocol record command-buffer driver
and GPU timestamps, using identical hooks and the same frozen qualified
metallib. Each process has three warmups and three retained samples. The
attribution probe is disabled; no extra GPU submission or completion callback
is added. Host logging can perturb timing, so these results are kept separate.

| Cold reference preparation | Baseline | Candidate |
| --- | ---: | ---: |
| Padded 4K driver | 40.455 ms | 41.754 ms |
| Planter 4K driver | 41.438 ms | 39.833 ms |
| Padded 4K GPU execution | 19.401 ms | 18.383 ms |
| Planter 4K GPU execution | 24.957 ms | 28.146 ms |

The diagnostic complete-call paired deltas are **+0.137 ms** for padded 4K and
**+1.694 ms** for planter, each faster in only one of three pairs. Warm reference
driver intervals are 0.03–0.055 ms. All nine retained warm encodes per arm/input
reuse all four arenas; every trimmed encode allocates fresh arenas. Warm tracing
does not show cache misses that would explain away its timing variation.

First-call samples from the fresh warm-protocol processes are also retained:
paired median deltas are −3.115 ms for padded 4K and +2.098 ms for planter.
These sparse, trace-enabled samples exclude backend construction and do not
establish a startup improvement.

The primary cold medians are encouraging, but their large paired ranges and
the separate diagnostic results prevent a reliable latency claim. The clear
result is lower allocation capacity with preserved output. A larger repeatable
cold improvement likely requires reducing more of the large backing arenas;
simply scheduling their first use earlier does not remove that work.

## Qualification and retained evidence

- Focused checks, including storage planning, AQ, Butteraugli, admission, and
  completed-frame reconfiguration: 8/8 pass.
- All 38 corpus inputs produce byte-identical codestreams against the committed
  AC-optimized baseline. Four decoded PFM pairs also match byte for byte; input,
  encoder, decoder, and output hashes are recorded.
- The final production suite passes **126/127**. The only failure is the
  inherited CPU `quantization_pipeline` golden mismatch: actual
  `0.24919039011001587`, expected `0.24914586544036865`, also recorded on the
  pristine baseline. `metal_completed_frame` passes without modifying its test.
- Production AQ, Butteraugli, and completed-frame tests pass **3/3** with Metal
  API and shader validation enabled. Three final production CLI encodes also
  match baseline bytes, including planter 4K.
- Results are recorded in `production-v2-tests.log`,
  `production-v2-validation.log`, and `production-v2-smoke/`; the final audit is
  `audit.json`. Production sources contain none of the diagnostic switches.

Evidence directories under `build/cold-preparation/`:

- `attribution2/`: first-use buffer attribution, valid independently of either
  prototype.
- `comparison-v2/`: primary untraced complete-call timings.
- `trace-v2/`: driver/GPU timestamps, cache events, and first-call observations.
- `parity-v2/`: 38 encoded pairs, four decoded pairs, and command/hash manifest.
- `candidate-v2-src/`, `candidate-v2/`, `production/`: measured source snapshot,
  measured build, and final production build.
- `rejected-five-plane.patch`, `rejected-five-plane-report.md`, `candidate-src/`,
  `candidate/`, `comparison/`, and `comparison-trace/`: rejected five-plane
  prototype. Its old report predates the full-suite rejection and must not be
  treated as qualification of that implementation.

Saved-data-only analysis:

```sh
python3 build/cold-preparation/analyze.py build/cold-preparation/comparison-v2
python3 build/cold-preparation/analyze.py build/cold-preparation/trace-v2
```
