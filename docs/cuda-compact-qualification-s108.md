# Compact ownership: quality, failure and batch qualification (S108)

Starting revision: `f934f9a`, following [S107](cuda-compact-frame-s107.md).
Work ran on 2026-09-07 evening local time / 2026-09-08 UTC, Windows,
CUDA 11.8, RTX 3060 Laptop. No encoder algorithm changes are combined with
this qualification. `GJXL_CUDA_COMPACT_AC` remains **OFF by default**.

The native representation now has broader exact-output, failure and concurrent
ownership coverage. Batch measurements support a useful memory reduction and
some whole-batch gains, but do not establish an unconditional speedup. The new
committed regression test injects allocation failures into frame assembly;
diagnostic campaign scripts and artifacts remain outside tracked production.

## Exact-output coverage

Two harnesses link the unchanged S107 dense/compact libraries. Each process
first encodes every request using an explicit reference backend. Dense
preflight writes non-overwriting byte and summary oracles; every later encode
checks those exact bytes and all deterministic summary fields. The reference
backend is destroyed before measured backends or the public batch driver are
created. No comparison substitutes checksums for byte equality. The evidence
validator also hashes inputs, libraries, executables, logs and frozen outputs.

| Campaign | Successful configurations or jobs | Exact top-level encode checks |
| --- | --- | ---: |
| Low-distance pilot | 6 configurations, dense and compact | 36 |
| Quality sweep | 15 inputs x 7 settings, dense and compact | 630 |
| Rate control | 4 inputs x 4 settings, dense and compact | 96 |
| High-range stress | 5 successful inputs, dense and compact | 30 |
| Batch preflight | 5 cohorts x 2 execution paths x 2 storage modes | 168 |
| Timed batches | 4 cohorts x 6 repetitions x 2 paths x 2 modes | 2,160 |
| **Total** | | **3,120** |

The fifteen inputs are four photographic contents at 500/1000/2000 size
labels, padded 1919x1079 and 3839x2159 fixtures, and the codestream sample.
The larger photographic derivatives are resampled versions, not twelve
independent photographs. Quality settings are distance/effort/final-score:
`0.1/1/off`, `0.1/10/on`, `1.2/4/off`, `1.2/7/on`, `4/1/off`, `4/10/off`,
and `16/7/off`. All three native widths are observed across the wider campaign.

Rate-control coverage uses sample, Flower 500, Keong 500 and 1080p. Each has
two five-attempt target-byte searches, one with final scoring, and maximum-error
limits 0.01 and 0.001. Target budgets are derived from the prior distance-1.2
codestream sizes, with a 1,000-byte floor. These are equality checks, not a claim
that every search reaches its requested budget. Maximum-error evaluation stays
on its intentionally dense non-policy path; its passing results do not prove
narrow maximum-error consumption. Search profiles report the largest AC owner
across attempts, not necessarily the selected attempt's width.

Pilot distances 0.1, 0.01 and 0.001 succeed for Flower and sample. Distances
0.0001 and 0.00001 reject in both builds with the same existing AQ-bound error.
Those eight rejected jobs are recorded separately and excluded from the 3,120
successful checks. They do not exercise the coefficient-width fallback.

### Real overflow decisions, synthetic inputs

Eight deterministic 65x67 grayscale float fixtures combine checker/ramp
patterns with amplitudes 1, 4096, 16,777,216 and 68,719,476,736 at distance 0.01,
effort 7. These deliberately high-range values are stress inputs, not ordinary
normalized photographs. Five inputs succeed. Three select int32 in the actual
compact resident policy: the amplitude-4096 checker and ramp, and the
amplitude-16,777,216 ramp. All nine compact full encodes match the corresponding
dense bytes and summaries. No flags, coefficients or width decisions are forced.
Both builds reject the other three fixtures with the same device-numerics flag
32; those six failed jobs remain separate from successful qualification.

This closes S107's missing complete-encode int32 fallback case, without claiming
that ordinary normalized images naturally need int32 at these settings.

## Allocation failure and GPU safety

`compact_frame_allocation_test` covers int8/int16/int32, owned/borrowed input,
and cached/recounted populations: twelve cases. It counts successful ordinary
C++ allocations, then fails each observed allocation individually. All 156
injected failures return OutOfMemory, preserve the input owner's exact pointer
and contents, and preserve the output frame's pointer, validity and codestream.
Each successful case also checks native width and an allocation-free native
group query while the next allocation is armed to fail.

Release and fully instrumented host-ASAN builds both pass all twelve cases,
156 failures and twelve allocation-free queries. This is host assembly coverage,
not GPU allocation-failure injection or a persistent system-wide OOM guarantee.
The first test version consumed its own dense fixture before constructing typed
inputs. Its invalid-input failure is retained; the corrected fixture borrows
that setup array. No production behavior was changed to make the test pass.

Five integrated Compute Sanitizer memchecks cover two int32-fallback inputs,
independent four-context mixed-quality encoding, the public four-worker batch
driver, and sample rate-control searches. They complete 42 additional exact
encode checks, each with an explicitly flushed completion marker, zero errors
and zero leaked bytes. These 42 are separate from the campaign total above.
S107's four packing-kernel sanitizer checks and detecting negative control
remain applicable to the unchanged kernel libraries.

The fresh default CPU build passes all 52 CTest tests, including the new
allocation test and install-consumer test. A fresh compact CUDA build passes
all 75 tests in one clean run, also including install-consumer. ASAN retains
the S107 diagnostic NOICF linker setting; production linker options are
unchanged. Other operating systems and GPU architectures are not qualified by
these Windows/sm86 runs.

## Concurrent timing and memory

Independent mode gives each request its own persistent backend and uses
concurrent asynchronous calls. Driver mode uses `VarDctBatchEncoder` and its
production backend lanes. Both check request order, per-item success, exact
bytes and summaries. Each request has a two-thread CPU budget. The four-image
cohort uses the four distinct 500-size photographs; larger cohorts contain two
copies of the indicated image. Preflight also covers four differing quality
settings in one batch. Two 4K images fit in both modes; four concurrent 4K
images are not retried, and no concurrency limit is raised.

Main timing runs 03:47:03.266995–03:52:40.010156 UTC (336.74 seconds). Six
repetitions alternate dense/compact process order and reverse cohort/path
order. Every process has one checked reference per request, two warm batches
and six measured batches. Thus there are 96 processes and 576 measured batches.
No builds, tests, sanitizers or telemetry samplers overlap timed processes.
Light source editing and natural machine-state drift remain limitations.

Outer time covers a complete batch, including internal frame cleanup and,
for independent mode, thread/future launch and join. It excludes input loading,
backend construction, returned-result destruction and comparisons. Reported
times are not per image and do not compare concurrency against serial encoding.

Each process contributes its median of six measured batches. Dense/compact
columns below are medians over six processes; the percentage is the median of
six adjacent-process ratios. These operations can disagree in direction, as
the driver 1080p row illustrates. None is a significance test.

| Cohort | Path | Dense ms | Compact ms | Paired median change | Six-pair range |
| --- | --- | ---: | ---: | ---: | ---: |
| Four mixed 500 | Independent | 43.65 | 42.69 | -0.98% | -8.24% to +11.83% |
| Four mixed 500 | Driver | 43.77 | 43.49 | -0.45% | -4.98% to +8.30% |
| Two Keong 2000 | Independent | 320.02 | 310.44 | -2.63% | -3.65% to +1.22% |
| Two Keong 2000 | Driver | 320.78 | 302.15 | -5.65% | -10.23% to -2.08% |
| Two 1919x1079 | Independent | 138.27 | 131.56 | -5.66% | -10.04% to -0.98% |
| Two 1919x1079 | Driver | 139.35 | 140.67 | -1.33% | -8.85% to +9.23% |
| Two 3839x2159 | Independent | 752.75 | 731.08 | -3.30% | -4.78% to -0.72% |
| Two 3839x2159 | Driver | 740.88 | 726.11 | -1.99% | -4.78% to +0.21% |

Only independent 1080p/4K and driver Keong 2000 improve in all six pairs.
All cohorts have negative paired medians, but that does not justify describing
every batch as faster or promoting a size cutoff from these data.

Two 4K frames' AC owners save 151.875 MiB in total. Observed paired median
process peak-working-set reductions are 152.89 MiB (independent) and 152.60 MiB
(driver). The corresponding 1080p reductions are 44.96/44.82 MiB and Keong 2000
72.62/72.53 MiB. Mixed-small driver working-set comparisons change sign.
Process peak includes initialization and prior reference/warm calls, not just
simultaneously live AC owners. Private-commit differences are mixed and noisy;
no consistent private-commit reduction is claimed.

## Disposition and next experiment

Keep the native compact implementation opt-in. Quality, concurrent ownership,
host allocation failure and real overflow selection now have direct evidence.
Small-batch timing and the public-driver 1080p variation still argue against
claiming a universal performance improvement. This is not optimization exhaustion.

The next compact-path experiment should test direct narrow AC-group packing.
Currently the GPU writes packed int32 groups, then rereads them to emit byte
and word payloads. Writing narrow payloads directly into the packed destination
could remove one pass; untouched quantized source could support a second dense
packing pass only on overflow. This is a code-inspection hypothesis, not an
implemented or measured gain. It must respect scratch lifetimes and independent
contexts. S106 composition/reduction and tile-scheduling candidates remain
separate; their isolated gains are not added to these results.

Scripts are ignored `build-cuda-ninja/profiles/s108_*`. Evidence is under
`U:/gjxl-cuda-diagnostics/s108`; the validator recomputes paired statistics,
checks accepted and rejected runs, verifies non-overlap, and anchors unchanged
S107 implementation libraries and all forty retained production runtime files.
No firewall/admin prompt is observed. No security, power, clock, priority,
driver or production-runtime setting is changed. The user's three untracked
Markdown files remain untouched.

```powershell
python build-cuda-ninja/profiles/s108_validate.py --frozen
```
