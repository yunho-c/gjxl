# Metal hardware-aware follow-up

Four measured optimizations are implemented on `perf/metal-kernel-dataflow`:
small-block metric reduction, DCT16 AC candidate fusion, fixed-shape Butteraugli
filters/Malta, and final EPF plus linear RGB conversion. The final implementation
and all qualification artifacts are recorded in the
[result ledger](../build/hardware-aware/evidence/final3/result-ledger.json).

Baseline: `62d617541471087cb29302f8a5abf87c96cd36da`. These are additional
changes after the earlier kernel-dataflow sprint; do not combine these ratios
with its results against `1f16769` without a new direct comparison.

## Complete-workflow results

Release builds, fully-resident Metal, distance 1.2, effort 7, on Apple M4 Pro
(20 GPU cores, 48 GiB), macOS 15.6, Xcode 26.3 build 17C529. Seven independent
process pairs alternate parent/candidate order, with three warmups and seven
measured calls per process. Times cover the complete public encoding call,
including CPU codestream work; backend setup and input loading are outside this
boundary. Instrumented GPU stage measurements are separate.

Times are medians of process medians. Changes are medians of paired percentage
changes, so they need not equal the ratio of the displayed medians. Negative
changes mean less time. Every raw sample and pair remains available.

| Workload | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 9.376 | 9.660 | +3.64% | 2/7 |
| Padded 1080p | 68.812 | 67.442 | -1.99% | 7/7 |
| Padded 4K | 225.678 | 219.220 | -2.94% | 7/7 |
| Kodak 17 | 22.388 | 21.993 | -1.54% | 6/7 |
| Planter 4K | 276.477 | 252.384 | -3.10% | 5/7 |

Padded 1080p and 4K improve in all seven pairs. Natural-image medians also favor
the candidate, with mixed pair directions. The 128x96 median regresses 3.64%
in the final cohort; substantial process variation means a small-image
regression cannot be ruled out. A separate focused 128x96 check
with five warmups and fifteen samples per process was flat (-0.05% paired
median); its EPF/conversion stage improved 15.75%. Keep that diagnostic cohort
separate from the final table.

Three paired stage cohorts, with two warmups and three samples per process,
compare the complete implementation to the same parent:

| Workload | All AC stages | EPF + conversion | All GPU stages, parent → candidate ms | All GPU stages change |
| --- | ---: | ---: | ---: | ---: |
| Padded 4K | -1.29% | -34.00% | 143.879 → 137.791 | -4.41% |
| Padded 1080p | -0.80% | -32.87% | 39.400 → 37.934 | -3.76% |

All-stage sums contain the measured GPU stages, not CPU time or complete-call
latency. EPF comparisons sum all EPF passes and conversion within each sample
before taking medians; comparing a fused pass against only an unfused pass
would use different work boundaries.

## Experiment dispositions

| Area | Selected implementation and evidence | Rejected or deferred variants |
| --- | --- | --- |
| Resident metric reduction | 64 lanes and explicit shuffle tail for 8x8 blocks only. 990 guarded bitwise cases. Isolated all-8x8 workloads improve about 17% at 4K; real mixed-transform metric stages improve only 0.76% at 4K and are flat at 1080p. | General replacement of larger-block trees; 64/128/256 shared/shuffle sweeps do not justify it. |
| AC candidate fusion | DCT16 forward, residual, inverse and weighted loss with concurrent channel ownership. Incremental DCT16 stages improve 5.47% at 4K and 3.87% at 1080p. CPU merge/ties and scalar cross-channel finalizer remain. | Sequential-channel fusion regressed 0.16% at 4K and 3.73% at 1080p and was removed. Naive DCT32 expansion needs 61,440 bytes, exceeding the queried 32,768-byte threadgroup limit. |
| Butteraugli blur/Malta | Fixed 16x64 one-output blur indexing and fixed 32x8 Malta indexing preserve arithmetic. 180 blur and 240 Malta guarded cases. Incremental Malta stages improve about 14%; blur plus Malta improve measured GPU stages about 2%. | Two/four outputs per thread and constant-address-space blur weights provided no extra advantage. |
| Final EPF/linear conversion | Direct and tiled final-pass variants write linear RGB immediately when resident Butteraugli is the only consumer. 168 guarded cases; exact serial/profiled comparisons for one/two/three EPF passes, with/without Gaborish. | Fusion into diagnostic or maximum-error paths, which still need filtered XYB. The initial remainder-based tiny-image sampler correction was replaced after a direct-kernel regression. |

Incremental EPF results compare the final bundle against the frozen `filters`
bundle (all earlier selections already enabled), with three alternating stage
pairs. The boundary includes all EPF passes plus conversion:

| Workload | Parent ms | Candidate ms | Paired change |
| --- | ---: | ---: | ---: |
| Padded 4K | 7.5958 | 4.9315 | -35.52% |
| Padded 1080p | 1.7296 | 1.1689 | -32.42% |
| Synthetic 128x96 | 0.0263 | 0.0224 | -15.31% |

Final isolated EPF timing uses five alternating pairs and seven measured GPU
submissions per side. Its parent runs the same direct/tiled EPF geometry followed
by the original conversion; this is kernel timing, not encode latency:

| Final pass | Geometry | Extent | Parent ms | Candidate ms | Paired change |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | direct | 512x512 | 0.1175 | 0.0972 | -16.13% |
| 1 | direct | 3840x2160 | 3.1851 | 1.7887 | -43.83% |
| 1 | tile32x4_p2 | 512x512 | 0.0643 | 0.0477 | -26.00% |
| 1 | tile32x4_p2 | 3840x2160 | 2.7323 | 1.3071 | -52.07% |
| 2 | direct | 512x512 | 0.0405 | 0.0218 | -47.32% |
| 2 | direct | 3840x2160 | 2.8916 | 1.3609 | -53.05% |
| 2 | tile32x4_p2 | 512x512 | 0.0414 | 0.0262 | -36.80% |
| 2 | tile32x4_p2 | 3840x2160 | 2.5594 | 1.0651 | -58.38% |

## Arithmetic and resource contracts

The 8x8 metric reduction drops zero-only upper levels and uses explicit
shuffle-down offsets 16, 8, 4, 2, 1 with the original tree parenthesization.
Larger blocks retain the 256-lane striped tree; there is no general SIMD-sum
reassociation.

DCT16 uses 192 threads (two SIMD groups per channel) and 15,360 bytes of static
threadgroup memory. All forward coefficients remain local. A threadgroup/device
publication barrier precedes X/B reads of Y coefficients. Original residual,
inverse, loss and scalar cost operation orders remain. Other transform families
retain the existing kernels. No AC candidate, tie rule or scratch allocation
is removed.

Blur retains its 16x64 footprint, 1024 threads, 33-tap arithmetic and 18,432-byte
tile. Malta retains 32x8 threads and a 2,560-byte halo. Generic and fixed Malta
wrappers share one arithmetic body. Probe-only sweep entry points remain
available, while production selects the measured variants.

Final EPF removes one three-plane filtered store/read and one conversion
dispatch per evaluated AQ iteration: 24 source-level bytes per pixel, not
measured DRAM traffic. Conversion retains gamma/cube/matrix/FMA order and error
bit 64. Normal EPF results retain sanitization/error bit 32; bypass pixels keep
the separate conversion's error behavior. Input/output strides stay independent
and filtering never aliases its input. Snapshot diagnostics and maximum-error
evaluation materialize filtered XYB; retained linear reconstruction remains
available. AQ iteration counts, quantization, policies and oracle tolerances
are unchanged.

The sampler also fixes an inherited invalid direct load for one/two-pixel
dimensions. Repeated reflection uses a constant result for size one and a
four-period bitmask for size two; larger dimensions retain their previous
formula. The first correction's variable remainder slowed direct kernels even
on larger images, so it was replaced and the final combination was requalified.
CPU-oracle cases cover 1x1, 1x9 and 2x2 for every EPF/Gaborish combination. Tiny
guarded comparisons use the parent's valid tiled kernel as control; larger
cases use the matching parent direct/tiled kernel.

Selections are gated on Apple GPU family 9 and applicable pipeline launch
limits; other families retain existing kernels. Qualification is on this M4 Pro
only. Family membership is not evidence of a speedup on another device.

## Setup and memory tradeoffs

Seven alternating process-cold pairs measure backend creation/preparation and
the first public call separately. Driver/disk caches are not reset; these are
not executable-launch or cold-driver-cache timings. `small` is 17x9 and `4k` is
3839x2159. Percentage columns use paired medians.

| Workload | Setup, parent → candidate ms | Setup change | First call, parent → candidate ms | First-call change |
| --- | ---: | ---: | ---: | ---: |
| small | 57.681 → 54.823 | -9.38% | 11.736 → 11.832 | +0.33% |
| 4k | 50.924 → 51.559 | +2.37% | 270.101 → 266.141 | -2.00% |

| Workload | Process peak, parent → candidate MiB | After trim, parent → candidate MiB |
| --- | ---: | ---: |
| small | 122.31 → 123.44 | 110.88 → 112.20 |
| 4k | 2780.56 → 2780.59 | 2051.94 → 2063.83 |

There are six additional prepared pipelines (DCT16, reduction and four EPF
variants). Blur/Malta replace prior selections. The final metallib grows from
1,999,580 to 2,208,188 bytes (+208,608; about 204 KiB); the encoder grows by
198,512 bytes. Setup results vary across cohorts: the earlier bundle measured
slower small-image setup, while this final cohort measured faster setup. These
observations do not establish a uniform cold-start gain or penalty. The final
small-process peak grows about 1.13 MiB. Existing image arenas and admission
planning remain: fixed single-image
managed backing peaks are unchanged, and concurrent peaks vary with scheduling.
No image-memory saving is claimed.

## Final qualification

- Both complete Release inventories pass 121/122 tests, with no skips and the
  same inherited `quantization_pipeline` failure: actual `0.24919039011001587`,
  expected `0.24914586544036865`. No tolerance changed.
- All 1,578 new guarded cases pass normally and under Metal API/shader validation:
  990 reductions, 180 blur, 240 Malta, 168 EPF/linear. Five integrated Metal tests
  pass with both validation layers enabled. Existing probes pass 1,344
  coefficient, 56 DCT and 432 EPF cases.
- Both revisions pass 22 pinned conformance fixtures. All 56 corpus/policy
  codestream pairs are byte-identical; three representative decoded pairs match
  through pinned libjxl `e8ff0976` (decoder SHA-256
  `9782c3474e41e5e5415da5fc68e7103d27047661385dc5b863dcad50ec4474ac`).
- Eight retained-resource scenarios pass changed-image, concurrent-caller,
  tight/full admission, shutdown and trim checks. All 24 retained codestream
  and decoded pairs match.
- Seven scoped C++ ASan/UBSan tests pass. Leak detection is disabled and the
  existing narrow `metal-cpp` null-call suppression is retained; shader
  correctness is checked separately by Metal validation.
- Five measurement-controller tests pass, including equivalent EPF/conversion
  boundaries and detection of active study children. `git diff --check` passes.

## Profiling, provenance and reproduction

A fresh parent Metal System Trace captured 3,354,416 counter samples overlapping
46 compute encoder windows. Resident AQ showed about 67% occupancy, a 64%
instruction-throughput limiter and a 10% L1-cache limiter. AC search showed
44% occupancy and a high launch limiter. These device counters can overlap
system GPU work and do not identify an individual shader's bottleneck.
Dispatch-boundary timestamps were unavailable; stage-boundary timing was
supported. Apple's [counter guidance](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
and [shader performance discussion](https://developer.apple.com/videos/play/tech-talks/10580/)
informed the experiments, not the numerical speedup claims.

Final evidence lives under `build/hardware-aware/evidence/final3`, and the frozen
Release runtime is `build/hardware-aware/candidate-final3`. The source archive
includes new files as well as tracked changes. `final-source-identity.json`
records source/runtime hashes, toolchain, submodule pins and artifact sizes.
`commands.json` records the exact final commands, times, exit codes and logs.
The reporter verifies hashes and recomputes summaries from retained raw samples.

The parent evidence directory retains incremental cohorts and rejected attempts.
Reduction timing that overlapped a counter export and EPF timing that overlapped
the independent libjxl study are excluded. The clean first bundle measurements
exposed the remainder-sampler regression and are superseded by `final3`, not
pooled with it. The user stopped the libjxl study before final measurements;
the runner checks competing work before and after every timed process.

`build/hardware-aware/evidence/run-final3.py` records qualification, sanitizer
builds and the measurement sequence; replay it only with new output/build
directories. Release configure/build logs are `final3-configure.log` and
`final3-build.log` in the parent evidence directory. The Release build uses
`GJXL_BUILD_TESTS=ON`, `GJXL_BUILD_BENCHMARKS=ON`, and
`GJXL_ENABLE_LIBJXL_REFERENCE=OFF`.
To verify the retained final result without rerunning benchmarks:

```sh
python3 tools/metal_dataflow/hardware_aware_report.py \
  --evidence build/hardware-aware/evidence/final3 \
  --output build/hardware-aware/evidence/final3/result-ledger.json
```
