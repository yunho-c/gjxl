# Butteraugli traffic study: measured headroom checkpoint

> Historical study checkpoint. The final disposition and commit status are in [REPORT.md](REPORT.md) and [README.md](README.md). Any pending work below refers to the time of that checkpoint.

The evidence demonstrates remaining achievable performance on the M4 Pro. Optimization exhaustion is not established. No production source has been changed; all prototypes live in the detached study checkout.

## Ordinary complete-call confirmation

Frozen `4f3e414` baseline (current-main shaders plus production-aligned profiling), M4 Pro 20 GPU cores / 48 GB, distance 1.9, eight CPU participants. Each row uses seven alternating independent-process pairs, three warmups and seven samples per process. Input loading/backend creation are outside the call; CPU serialization and returned codestream production are inside. All paired outputs and per-process repeated calls are byte-identical and submission counts match.

Percentages are medians of paired percentage changes. Negative means less time. Separate experiments are not a direct comparison between candidates, and the controls are not a fixed correction.

| Candidate vs original baseline | Alpine Lake 24 MP, e7 | Forest Stream 48 MP, e10 |
|---|---:|---:|
| Identical-binary control | +0.401% (3/7 faster) | -0.430% (6/7 faster) |
| Main DC producer/reducer fusion | -0.508% (5/7) | -0.947% (6/7) |
| Branch-free adjacent-load low/medium filter | **-2.939% (7/7)** | **-1.530% (6/7)** |
| Main+sub DC fusion and that filter | **-2.538% (7/7)** | **-2.504% (7/7)** |

The sub-percent DC-only differences remain difficult to distinguish from drift. The larger filter and combined results provide stronger selected-workload evidence. This is not yet a broad production qualification: more contents, targets, policies, decoded checks and smaller controls remain. The combined candidate is not established as superior to the filter-only candidate by these separate cohorts.

## What changed and why

The original 33-tap low/medium filter already used direct loads and a 16x64 output tile. Twelve straightforward adjacent-load prototypes were slower. Removing their per-tap output predicates for interior pixels changed the result: two adjacent horizontal outputs share source loads, retain independent ascending accumulation order, and use two vertical outputs per thread. Eight refinements passed guarded bitwise checks. The best isolated results improved about 19-21%; independent integrated low/medium stages improved about 21-22%. A 24 MP control in the first refinement screen drifted +4.98%; retain that anomaly and rely on the independent integrated/ordinary evidence for the encoder claim.

DC fusion calculates the exact reference/distorted low-frequency error where the distorted low frequencies are produced, then writes one masked DC value instead of three distorted low planes. The main reducer reads that value instead of six reference/distorted low planes. The naive combined source-level accounting saves about 12 bytes/pixel after charging earlier reference/mask reads; this is not measured external-memory traffic. The main reducer improved about 24% and the affected filtering+reduction scope improved about 8-9%. Applying the same representation to subscale reduced its mask/final stage about 17-20%, with mixed incremental instrumented whole-call results.

A further AC prototype moves exact per-channel L2 composition into the final Malta accumulation. Against main+sub DC fusion, it reduced main resident reduction about 49% and sub-final 32-34%, while increasing Malta about 8-8.5%. Counting filtering, reduction, sub-final and Malta together gives a 6.3% improvement at 24 MP and 8.2% at 48 MP. These are three-pair stage results; ordinary AC confirmation is pending. AC fusion relocates reads/arithmetic and improves their access context; the removed reducer reads must not all be counted as removed DRAM traffic.

## Negative and unresolved evidence

- Ninety isolated variants have completed initial timing: 14 tile geometries, eight rolling tiles, 12 conditional adjacent-load layouts, 12 normalization variants, eight branch-free refinements, 24 short-filter layouts, and 12 Malta geometries. Raw results and controls remain in their family directories.
- Taller/rolling 33-tap tiles improved isolated kernels but had mixed integrated behavior. Normalization reuse added no demonstrated gain over its geometry. Conditional adjacent-load layouts regressed.
- The new direct-load 7-tap filters improved about 23-34% in isolation. All 13/15-tap versions were slower, including their best variants. A disjoint-input 7-tap ultra/mask implementation is being qualified; raw blur timing does not include its full epilogue.
- Malta's 32x16/two-output tile improved an equal-weight LF/full isolated sum about 3%. Its actual six-pass stage composition and complete encoding still need testing.
- Exact mask-curve caching is being screened optimistically at consumers, excluding cache construction and additional storage. A consumer win would still require end-to-end qualification; a flat consumer result argues against paying those costs.
- Vertical filter reuse, adjacent Malta outputs, remaining reduction layout/materialization choices, broader contracts and final combinations remain open. There is no basis to declare optimization exhaustion.

The primary checkout status remains unchanged. Commands, binaries, shaders, source patches, input hashes, raw samples, exceptions and resumable states are retained in this directory. See `PROGRESS.md`, `OPPORTUNITIES.md`, `TRAFFIC-NOTES.md`, `INCIDENTS.md`, and `experiments.jsonl`.
