# Effort-three quantization policy

Ordinary effort 3 retains the initial per-block AC quantizer and fixed Y
quantization thresholds (`kFixedRawQuant`). It also enables adaptive DC smoothing
by default. The existing explicit smoothing option overrides that default
independently of AC coefficient coding.

Ordinary means default density, distance or target-size control, and any supported
GPU mode except maximum throughput. High density, maximum-error control, and
maximum throughput keep their previous e3 policies. The entropy compression
setting is independent: automatic and maximum compression use the same frontend
rule. Other efforts keep their existing AC decisions and smoothing defaults.

The rest of ordinary e3 remains fixed DCT8, uniform initial quantization,
Gaborish off, two EPF passes, zero perceptual-refinement updates, ordinary DC
rounding, and zero extra DC precision. Explicit DC controls remain available.

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

## Retained rate evidence

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
changes. Fixed Y thresholds remain GJXL's 0.58/0.64, rather than libjxl e3's
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

## Integration validation

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
