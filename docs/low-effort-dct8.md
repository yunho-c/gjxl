# Low-effort frontend policy

Ordinary efforts 1–4 select DCT8 for every 8×8 base block without invoking
AC-strategy search. They disable Gaborish and initialize every quantization
block to `0.79 / distance`, matching libjxl’s e1–4 initialization boundary.
Effort 5 enables mixed-transform search, spatial initialization, and Gaborish.
Ordinary effort 4 now skips perceptual AQ refinement, like efforts 1–3.
It also enables the [default DC integer-mapping search](entropy-defaults.md)
under automatic compression; native context-map compression applies at every
effort. These serializer choices preserve the frontend's reconstruction.
The adaptive-quantization update schedule is:

| Effort | Transform selection | AQ updates |
| --- | --- | --- |
| 1–4 | Fixed DCT8 | 0 |
| 5–6 | Mixed-transform search | 1 |
| 7 | Mixed-transform search | 2 |
| 8–9 | Mixed-transform search | 3 |
| 10 | Mixed-transform search | 4 |

The high-density and maximum-error overrides retain their existing search
and refinement behavior; high density still requests four AQ updates.
Target-byte and target-bpp retries follow the ordinary effort
policy. Maximum-compression controls the serializer independently and does
not re-enable AC search. The dedicated maximum-throughput path keeps its
existing policy. The low-level quantization pipeline defaults to mixed search;
the encoding workflow explicitly selects the effort policy.

The shared CPU/GPU orchestration builds a complete DCT8 grid before calling
adaptive quantization and never calls the AC provider in fixed mode. This
removes candidate preparation, transforms, scoring, readback, and CPU merging.
Resident preparations release retained AC-search scratch when switching to
fixed mode. CPU, resident Metal, and compatibility Metal admission plans omit
search-only storage and include overlap for replacement strategy grids.

Initial quantization and CfL remain. Ordinary efforts 1–4 use a direct uniform
fill on CPU and Metal, skipping spatial gradient, erosion, modulation, and
mask convolution work. Their inverse-Gaborish stage is disabled and the frame
header explicitly signals Gaborish off while retaining two EPF passes.
Serializer admission includes the longer nondefault filter header. Ordinary
effort 4 uses zero AQ updates in the shared execution and admission policy.
This changes its output and perceptual tradeoff; the qualification below
distinguishes the policy change from the separate diagnostic kernel changes.

## Resident search-data omission

Encoding with fixed DCT8 now prepares the Metal frontend without its initial
AC-search masks. The gradient and erosion kernels omit mask-only arithmetic
and stores; the pixel-mask convolution and validation dispatches are skipped.
The three mask planes use separate minimal argument bindings instead of
full-sized backing. Initial strategy-mask and CfL host readbacks are also
omitted. Uniform initial quantization and device CfL still feed final
coefficient coding; higher-effort spatial paths retain inverse Gaborish.

This applies to zero-update efforts 1–4. Full
initial-quantization diagnostics retain all masks and initial CfL outputs;
efforts using mixed-transform search retain their existing preparation. The
cached evaluator includes this choice in its compatibility check, so changing
between encoding, diagnostics, and mixed-transform search rebuilds the
appropriate preparation. Device and profile storage plans follow the same
policy. Shared host preparation bounds remain conservative.

## Validation

`low_effort_strategy_policy` checks the effort/override matrix, AQ update
counts, provider bypass, mixed/fixed preparation reuse, and reduced storage
plans. It uses stub providers and performs no image encoding or GPU work.
The CPU, resident, and compatibility workflow storage tests support
`--plans-only` for their allocation-free plan matrices (and mock policy or
search-interval checks).

The workflow regression test also checks DCT8-only summaries on CPU and Metal
at efforts 1–4, includes effort 5, and checks scored/unscored byte parity
through effort 4. The GPU pipeline regression covers switching mixed/fixed
preparations, zero candidate statistics, and scored/unscored fixed-policy
byte parity.

Release compilation and the non-encoding checks passed: the new policy test,
4,480 CPU plan shapes, 2,240 resident plan shapes, and 2,800 compatibility plan
shapes, plus their mock-policy and search-bound checks. The resident plan test
now includes score/timing publication storage in its aggregate bound comparison;
without AC search, it cannot assume a strict saving from releasing search
storage before completion.

After encoder runs were enabled, the Release workflow, low-effort policy,
CPU/resident/compatibility storage-plan, and Metal quantization-pipeline tests
passed. The resident retry-lifetime assertion now expects no AC-search storage
for fixed DCT8 and retained search storage for mixed-transform preparations.
The three previously calibrated DCT8 probes also match the production e1
output bytes exactly. These checks do not establish corpus-wide rate-quality
behavior or speed parity.

The search-data omission tests compare initial quantization, final CfL, and
encoded bytes against the full preparation. They exercise diagnostic/encoding
cache transitions with zero and one AQ update, forbidden mask/CfL outputs,
and atomic numeric/readback failures. Metal API and shader validation also
pass on the affected workflows. This establishes output parity and less
search-only backing; measured complete-encode timings remain mixed.

## Uniform-field qualification

The September 13, 2026 study used six Kodak, six CLIC, and three 12 MP
photographs, with pinned fast SSIMULACRA2 and no extrapolation. Gaborish off
plus uniform initialization, retaining e4’s one AQ update, changed mean
BD-rate relative to the previous e4 by −4.93% at scores 75–85, −5.18% at
45–55, and −4.21% at 25–35. The lower ranges had 2/15 and 3/15 per-image
regressions respectively, with worst increases +2.36% and +4.86%. These
content-dependent regressions were accepted for the new default. This is
not a claim of improvement on every image, metric, or resolution.

The original measurements and decoder/hash audits are retained in
`/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/e4-rca-20260913`
and its sibling `e4-low-quality-20260913`. Those studies evaluated rate,
not production latency. The production replay and validation are recorded
in `low-effort-frontend-qualification.md`.

## Effort-4 zero-update default

The September 14, 2026 investigation isolated zero AQ on revision `b1fbfc1`
while keeping the existing e4 DC, smoothing, transform, and serializer
policies. Seven rotated repetitions at six measured SSIMULACRA2 points near
85 reduced complete encode time by 30.3–61.3% on the M4 Pro. Across all 65
original images and seven quality settings, zero AQ changed mean BD-rate by
+0.17% versus baseline GJXL over scores 75–85. Six secondary-metric checks
found standard Butteraugli error changes from -0.74% to +10.21% despite
matched SSIMULACRA2. This perceptual tradeoff was accepted for the e4 default.

Only the zero-update policy is promoted. The DC-fusion, final-CfL packing,
and constant-field-statistics patches remain diagnostic. In particular, the
full-corpus speed result for their combined build is not a timing result for
this policy-only change. The frozen measurements are retained locally under
`build/e4-study/`, with the detailed investigation in the local artifact
`docs/e4-speed-investigation.md`.

The policy-only Release build passes 11 focused CTests covering effort and
override policy, CPU/Metal workflow and scoring parity, DC policy, Metal
quantization, CPU/resident/compatibility storage, admission, and independent
decoder smoke. Six representative encodes through 48 MP are byte-identical
to the qualified zero-AQ diagnostic build. The fresh build and logs are in
`build/e4-default/`; the frozen investigation binaries remain unchanged.
