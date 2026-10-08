# Low-effort AC quantization policy

Ordinary efforts 1–4 retain the initial per-block AC quantizer and fixed Y
quantization thresholds (`kFixedRawQuant`). DC settings remain independent:
ordinary rounding and no extra precision at e1–3, prediction-aware quantization
with one extra precision bit at e4, smoothing off at e1–2 and on at e3–4.
Explicit DC and smoothing options override their defaults without re-enabling
AC adjustment.

Ordinary means default density, distance or target-size control, and any supported
GPU mode except maximum throughput. High density, maximum-error control, and
maximum throughput keep their previous policies. The entropy compression
setting is independent: automatic and maximum compression use the same frontend
rule. Efforts 5–10 keep their existing AC decisions and smoothing defaults.

Ordinary e1–4 remain fixed DCT8, uniform initial quantization, Gaborish off,
two EPF passes, and zero perceptual-refinement updates. The e4 DC integer-mapping
search remains enabled under its existing policy.

## Implementation

`ResolveAcCoefficientDecision` selects the mode once in the encoding workflow.
`AdaptiveQuantizationOptions` carries it into CPU coefficient coding, exact GPU
coefficient coding, and resident/frame-only GPU preparation. GPU preparation
reuse includes the coefficient decision mode in its identity, including the
fast matrix-scale reconfiguration path.

Metal and CUDA skip AC-adjustment dispatches for fixed raw quantization and use
the existing fixed-threshold coefficient kernels. Resident quantizer construction
still supplies the authoritative raw field. Metal frame publication consumes
that device field even when adjustment is disabled. Storage plans retain their
conservative adjustment-scratch allowances.

The C ABI and Rust option layout are unchanged. The CLI
`--no-adaptive-dc-smoothing`, C `GJXL_DC_SMOOTHING_DISABLED`, and Rust
`adaptive_dc_smoothing: Some(false)` disable smoothing without re-enabling AC
adjustment.

## E1, e2, and e4 extension

The October 7, 2026 follow-up studies used frozen base `1f068e1` and changed
only the ordinary effort's coefficient-decision mode. Each study kept its DC
quantization and smoothing defaults. All 65 paper images and seven quality
settings were replayed against native controls; all native outputs reproduced
the retained baseline exactly. Per-image PCHIP log-rate integration over
SSIMULACRA2 75–85, equally averaged without extrapolation, gives:

| Effort | Before / same-effort libjxl | AC adjustment off / libjxl | Off / native GJXL |
| --- | ---: | ---: | ---: |
| 1 | −7.086% | −9.143% | −2.224% |
| 2 | −7.086% | −9.143% | −2.224% |
| 4 | −0.214% | −1.332% | −1.118% |

The direct candidate/native values are separate per-image comparisons, not
differences between the libjxl-relative averages. Each effort improves 63/65
images; the worst regressions are +0.868% at e1/e2 and +0.834% at e4. E1/e2
produce identical codestreams at every tested setting in each arm. Akima gives
−2.205% and −1.092% for the direct e1/e2 and e4 changes, respectively.

At matched SSIMULACRA2 near 80, the e1/e2 candidate reproduces the earlier
no-smoothing e3 study's +2.312% conventional Butteraugli error on six accepted
pairs, verified by exact encoding and independent decoding replay. Fresh e4
calibration gives −0.611% Butteraugli error on six accepted pairs out of twelve;
the other six pairs remain unresolved under the fixed tolerance and probe
budget. These geometric means apply only to their accepted diagnostic subsets,
not to the whole corpus or to Butteraugli BD-rate.

Artifacts, manifests, raw results, and audits are retained under:

- `/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/e1-e2-ac-adjustment-20261007/`
- `/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/e4-ac-adjustment-20261007/`

These rate checks ran on battery and establish no timing claim. The production
policy preserves the qualified recipes, including smoothing off at e1/e2.

### E1–4 integration validation

The combined policy is validated in a separate Release build at
`build/e1-e4-integration/`, with logs, source and binary hashes, and output
records in `build/e1-e4-integration-qualification/`:

- All 170 native CTests pass, including low-effort mode boundaries, DC overrides,
  C API behavior, and CPU/exact-Metal parity across e1–4.
- All 11 Rust tests pass, together with formatting and Clippy. The Rust DC
  override test now includes e1/e2.
- The GPU quantization pipeline passes with Metal API and shader validation.
- All 1,820 ordinary e1–4 outputs (65 images × seven settings × four efforts)
  exactly reproduce the qualified study codestreams. Each setting runs a
  validation encode and a second byte-identical encode. Thus the retained rate
  results apply to the combined production policy on the same corpus and
  metric. E3's existing smoothing-on recipe is unchanged.
- Thirty-six independent pinned-libjxl decodes reproduce the retained linear
  float hashes at Q30/Q80/Q95 on two Kodak images and one CLIC image.
- Twenty-eight CPU/Metal controls at e3 and e5–10, plus fourteen high-density,
  maximum-error, and maximum-throughput controls, remain byte-identical to the
  pre-extension implementation.

The older qualification builds and study artifacts are preserved. CUDA
compilation/runtime remain untested; the separate Metal timing qualification
is recorded below.

### E1–4 timing qualification

AC-powered paired timing against frozen parent `1f068e1` passes for the
integrated policy. The same retained executables were used in an 18-image
experiment and a separate six-image confirmation under stricter background-load
limits. Both use Q30/Q80/Q95, eight participating CPU threads, ordinary fully
resident Metal, and four balanced before/after process pairs per setting.
Each process performs one validation encode, two warmups, and five timed
complete public encode calls. All outputs match the qualified codestreams.

| Effort | 18-image encode time change | Six-image idle confirmation |
| --- | ---: | ---: |
| 1 | −4.30% | −5.72% |
| 2 | −4.27% | −3.97% |
| 3, unchanged control | −0.58% | +0.82% |
| 4 | −3.77% | −3.62% |

These are equal-setting geometric means of median paired time ratios; negative
means faster. The cohorts are reported separately, not pooled. All changed
efforts improve with conditional 95% interval upper bounds below no change,
no per-image mean slowdown above 5%, and no per-setting median slowdown above
10%. E3 remains within its ±3% aggregate and ±10% per-image control limits.
The six-image subset, confirmation protocol, and agreement limit were declared
before inspecting the first experiment's timing results. Its same-subset
replication check also passes.

The two experiments contain 1,152 accepted process pairs and 11,520 timed calls.
Seven pairs were excluded for monitored background load and retried; no
exclusion used measured latency. The broader run began with media analysis
active and recorded its later authorized pause. The confirmation retained the
original stricter admission limits, with media analysis and Core Spotlight
temporarily paused under restoration watchdogs. Both services were restored.

Current e1/e2 times are within about 1.2% of e3 in these experiments, so the
earlier large apparent e3 advantage over e1/e2 does not persist here. These
same-distance, warm single-image measurements supplement the rate evidence;
they do not refresh full-corpus paper throughput, establish matched-quality
speed, or qualify CPU/CUDA, batch, or cold-start timing.

The combined report and audit are in `build/e1-e4-timing-qualification/`.
Complete manifests, binaries, source patches, samples, attempts, environment
records, and restoration logs are retained separately in
`build/e1-e4-timing-desktop-qualification/` and
`build/e1-e4-timing-idle-confirmation/`.

## Original e3 rate evidence

The October 7, 2026 attribution study used frozen GJXL `d71eeb0`, libjxl
`e8ff0976`, and all 65 original paper images. Arithmetic means of per-image
SSIMULACRA2 BD-rates over 75–85, with complete coverage and no extrapolation:

| AC adjustment | DC smoothing | BD-rate versus libjxl e3 |
| --- | --- | ---: |
| Enabled | Off | +1.670% |
| Enabled | On | +1.416% |
| Disabled | Off | −0.606% |
| Disabled | On | −0.687% |

The measured bypass includes both raw-quant adjustment and adaptive Y-threshold
changes. Fixed Y thresholds remain GJXL's 0.58/0.64, rather than libjxl e1–4's
0.56/0.62. This implements the tested candidate, not exact libjxl arithmetic.

The SSIMULACRA2 gain has an accepted cross-metric tradeoff. At actual matched
SSIMULACRA2 near 80, the smoothed candidate increased conventional Butteraugli
error by 2.37% versus native GJXL on seven accepted pairs, and by 3.27% versus
smoothing-only GJXL on eight accepted pairs. These are geometric means on
pair-specific subsets, not Butteraugli BD-rates or percentage losses of perceived
quality. The no-smoothing candidate's +2.31% uses a different six-pair subset.

Retained study root:
`/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/e3-policy-attribution-20261007/`.
The original and `no-smoothing/` reports, manifests, codestreams, scores, and
audits remain authoritative for the historical measurements.

## Original e3 integration validation

Integration builds and controls are retained in `build/e3-qualification/`.
The base checkout is `67caa6d`; the baseline binaries were built before editing.
Validation uses an independent build tree at `build/e3`, not the study's build
or binaries. Replay records identify source hashes, binary hashes, and each
expected and actual codestream hash.

Completed checks:

- All 455 default outputs (65 images × seven settings) exactly reproduce the
  study's smoothing-on/fixed-raw codestreams. Thus the retained −0.687%
  same-effort result applies to these production outputs and the same inputs,
  decoder, metric, and interval; no new score interpolation was required.
- All 84 smoothing-off outputs and 84 unmodified-baseline outputs reproduce
  their respective study controls. All 65 source PFM hashes were verified.
- Twenty-four before/after controls at e1, e2, e4, e5, e7, and e10 on CPU and
  Metal are byte-identical. Five additional controls cover high-density,
  maximum-error, and maximum-throughput behavior. High density uses its CLI
  preset; combining an explicit effort with that CLI flag is rejected.
- Nine independent pinned-libjxl decodes at Q30, Q80, and Q95 match the study's
  linear-float PFM hashes exactly, across two Kodak images and one CLIC image.
- Thirteen focused CTests pass, covering policy scope, explicit smoothing
  overrides, C API behavior, batch encoding, CPU/exact-GPU coefficient parity,
  GPU reuse across decision-mode switches, reconstruction, and storage plans.
  The Metal pipeline suite also passes with API and shader validation enabled.
- The Rust automatic-DC/default-smoothing override test passes against a fresh
  native build. CUDA resident dispatch and regression coverage were updated,
  but CUDA compilation and execution were not available on this Mac.

The replay scripts, ledgers, progress, input/binary manifest, unchanged-output
controls, decoder controls, and test logs are in the integration artifact
directory. Initial test runs exposed stale smoothing expectations and test
harness mode assumptions; their logs are preserved alongside passing reruns.

Timing qualification is excluded because a concurrent laptop compression
workload was active. The rate study also made no throughput claim.

### Combined revision with current main

Merge revision `6c880ad` combines the e3 policy with main `e1e4cc6`, including
the ordinary-e5 zero-refinement policy. Both revisions were built in separate
Release build trees; the original qualification binaries were preserved.

- All 170 native CTests pass, with no failures or skips.
- All 11 Rust workspace tests pass, together with formatting, Clippy, and the
  native incremental-rebuild check.
- The GPU quantization pipeline passes with Metal API and shader validation.
- Thirty-six e3 codestream checks cover CPU and Metal, both smoothing settings,
  and Q30/Q80/Q95 on two Kodak images and one CLIC image. Metal matches the
  retained study hashes; CPU matches the pre-merge e3 candidate. Nine independent
  libjxl decodes also match the study's linear-float hashes.
- Thirty-six non-e3 controls cover every other effort on CPU and Metal at two
  distances. These and five specialized-policy controls are byte-identical to
  current main, including its intentional e5 change.

The scripts, commands, source and binary hashes, test logs, and final audit are
retained in `build/e3-integration-qualification/`. The unrelated dirty main
worktree's status and staged/unstaged patches were verified unchanged. CUDA
build/runtime and timing qualification remain outside these completed checks.
