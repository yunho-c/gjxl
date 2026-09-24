# CUDA integration policy qualification

The integration adopts main's effort and writer policies. The old CUDA branch
is a useful performance and quality baseline, but its output is not the target
for deliberate main policy changes. This record separates those changes from
backend drift. Automatic CUDA selection remains opt-in.

The main baseline is `bb7b714e858ffe7668a84c8588745a99b7015f39`; the old CUDA
baseline is `6e8277b069b79d7ef48b05514f8a160eb6f23648`. Both are ancestors of
the integration. The shared policy functions retain main's conditions and
iteration counts; the GPU mode field is renamed and the final-CfL accelerator
selection also admits CUDA. In particular:

- Ordinary efforts 1–4 use fixed DCT8, uniform initialization, Gaborish off,
  and zero perceptual updates. Explicit density/error modes retain their
  separate recipes. See [main's low-effort qualification](low-effort-dct8.md).
- Effort 8 inherits rate-optimized entropy search with a complete-stream
  balanced fallback and nonlinear resident final CfL. Efforts 8–10 use the
  shared eight-step CfL policy; effort 10 enables dense DCT32-family placement.
  See [writer qualification](e8-rate/integration.md) and
  [CfL qualification](e8-rate/cfl-integration.md) for the original studies.
- DC prediction, precision, smoothing and context-map policies follow main.
  CUDA's native coefficient ownership and serializer transport optimizations
  remain in place.

## Decoded-quality comparison

Four hash-verified photographs (flower, keong, riaphotographs and bliznaca)
were encoded at efforts 1/4/7/8/10. For each old-CUDA distance 1.2/3 point,
external distance bracketing selected the integrated stream nearest its byte
size, with at most 18 attempts. This is a quality experiment, not the public
target-size controller. The size gate is ±0.2%; 36 of 40 comparisons meet it.
The four unmatched points are excluded from the table, not extrapolated.

Independent decoding and Butteraugli use libjxl revision
`e8ff09762481785938d8e4e01333ed3917571161` and linear-sRGB float input/output.
Ratios below are integrated/old-CUDA decoded Butteraugli error; lower is better.
They are individual matched-size comparisons, not BD-rate measurements.

| Effort | Matched pairs | Median error ratio | Range |
| --- | ---: | ---: | ---: |
| 1 | 6 | 1.1834 | 0.8451–1.2781 |
| 4 | 6 | 1.2739 | 1.0378–1.5478 |
| 7 | 8 | 1.0293 | 0.9986–1.1286 |
| 8 | 8 | 0.9955 | 0.9212–1.0248 |
| 10 | 8 | 1.0045 | 0.9782–1.0510 |

All 40 selected resident outputs reproduce byte-for-byte with the production
exact-arithmetic build (`ab22c89`; replay source revision `24e25cc` changes only
Rust rebuilding and CI). Consequently their retained decoded measurements
still apply. Sixteen additional CPU controls use the identical selected
distance and effort for the low-effort points. Every control has exactly the
same reported Butteraugli score as resident CUDA; 15/16 streams are identical,
and the remaining size difference is 0.0287%. This supports attributing the
material low-effort quality shift to the shared policy rather than CUDA AQ
arithmetic on these samples. It does not establish universal CPU/resident
byte equality or prove that every image benefits from the new policy.

The integration retains the accepted main defaults. The low-effort error
increases and higher-effort outliers remain visible; they are not described as
quality improvements. Main's larger SSIMULACRA2 studies and this small
Butteraugli comparison use different metrics and populations, so their
percentages must not be combined.

## Performance and accounting boundary

The initial whole-call default-policy screen found effort 1 faster on odd
1080p/4K images (76.7→41.8 ms and 178.6→123.1 ms), while efforts 7 and 8
became slower. That screen has only five retained samples per point and uses
fixed distance rather than matched decoded quality. It establishes neither a
general speedup nor the cost per unit of quality. Effort 8 now performs more
AQ, nonlinear CfL and writer search, so its old-branch latency is not a
same-work comparison.

Separate repeated, alternating whole-public-call controls with legacy DC
isolate integration overhead more closely: small, odd 1080p/4K, photograph,
and two-caller mixed-batch median paired ratios range from 0.936 to 0.987.
Finite-budget pressure runs exercise 4K and concurrent batches with CPU caps
and tight/full reservations. Managed domains return to zero after trim.
The managed-backing contract excludes driver-internal overhead; device-wide
free-memory snapshots are not per-process GPU peaks.

The subsequent exact-arithmetic correction costs about 3.5% at exact effort 8
on two photographs against its immediate parent. The resident controls do not
show a material change. See [the exact arithmetic record](cuda-exact-arithmetic-integration.md)
and [profiling controls](cuda-profiling.md) for those separate interventions.
All timings are from one RTX 3060 Laptop GPU. No Metal/CUDA speed comparison
or performance guarantee for another GPU follows from them.

## Retained local evidence

Under `build/integration-evidence/`:

- `photo-quality/`: 160 original encode/decode/metric points and tool/input hashes.
- `photo-bracketed-size/`: all 40 searches, every attempted size and selection.
- `policy-quality-replay/`: current hashes, 16 CPU controls, commands and scores.
- `performance-screen/`: default-policy timing screen.
- `memory-performance/`: 40 paired timing and eight finite-budget processes.

The [integration work record](cuda-integration-progress.md) tracks the
remaining platform and acceptance gates.
