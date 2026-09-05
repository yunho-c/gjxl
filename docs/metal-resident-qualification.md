# Metal resident quality regression and rollout qualification

The default regression run encodes fully resident GJXL and pinned libjxl at the
**same requested distance**, independently decodes both, and measures their
quality and size. Equal knob settings do not imply equal quality. The normal
run performs no quality-matching search and never requests GJXL CPU encoding.
A separate, opt-in matched-quality mode retains the bounded reference search.

Initial runs collect observations. Subsequent regression runs check Metal's
fixed-distance quality and size against an explicitly accepted Metal baseline.
No comparison mode changes automatic backend selection or accepts its own
initial output as a baseline.

## Run

Initialize recursive submodules and install Python 3, CMake, Ninja, and
ImageMagick. Actual qualification requires the currently qualified Apple M4 Pro.
The Python decision tests do not require a GPU or reference binaries.

```sh
python3 tests/metal_resident_qualification_test.py
just metal-resident-qualify compact
just metal-resident-qualify full
# Optional, slower matched-quality comparison on the compact corpus:
just metal-resident-qualify compact matched-quality
```

Both Just recipes use an isolated Release build under `build/release`. Their
CMake targets build statically linked, pinned `cjxl`, `djxl`, and
`butteraugli_main`; record their binary hashes; prepare the corpus; and run the
suite. Same-distance is the default; set `--comparison matched-quality` on the
standalone runner, or `GJXL_METAL_QUALIFICATION_COMPARISON=matched-quality` in
CMake, for the slower search. The Just recipe explicitly selects its comparison
argument even when a build directory previously used the other mode. Initial
observations return a nonzero exit status. These targets are opt-in and are not
build dependencies of ordinary tests.
Full preparation downloads the hash-pinned photographic sources unless an
existing canonical pilot manifest is supplied:

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_METAL_QUALIFICATION_PILOT_MANIFEST=/absolute/path/to/pilot/manifest.json
cmake --build build/release --target metal-resident-qualify-full
```

The manifest may reside outside this checkout; inputs are shared read-only and
verified against the checked-in canonical hashes before every run. New corpus
preparation writes atomically and never overwrites an existing corpus. Changed
conversion results require an explicit review of the pinned input hashes.
`gjxl_encoding_benchmark --source-output` exports the exact built-in padded
stress sources before creating a Metal backend or running measurements.

Set `GJXL_ENABLE_METAL_QUALIFICATION_TEST=ON` to register the prepared compact
integration with CTest. Same-distance testing requires an accepted compact
baseline. Matched-quality testing accepts either a compatible baseline or an
explicit `GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO`. Configuration fails
without the applicable acceptance criterion. Build `gjxl_pinned_quality_tools` and
`metal-resident-prepare-compact` first; CTest does not build or download its
dependencies. The regular `metal_resident_qualification_driver` unit test is
always registered when project tests are enabled.

## Coverage

The compact corpus contains flat, gradient, hard-edge, saturated-primary,
textured, and seeded-noise fixtures, with aligned and padded geometries, plus
128x96 Flower and grayscale-patches crops from pinned libjxl testdata. The full
corpus adds 24 Kodak images, six imazen-26 photographs at 1080p and 4K, and two
padded stress inputs. Recipes, licenses, source hashes, and canonical pixel
hashes are retained under `tests/metal_qualification/`.

| Suite | Inputs | Distances | Policies | Cases |
| --- | ---: | ---: | ---: | ---: |
| compact | 8 | 12 | 11 | 1,056 |
| full | 46 | 32 | 11 | 16,192 |

Compact distances are `0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.499,
2.5, 2.501, 3.0`. Full distances cover 0.5 through 3.0 in 0.1 steps, plus
`0.999, 1.001, 1.199, 1.201, 2.499, 2.501`. This tests the interior as well as
the existing selection and quantization-matrix boundaries; sampled coverage is
not a mathematical guarantee for every real-valued distance.

The eleven policies are ordinary efforts 1, 4, 7, 8, 9, 10 and
maximum-compression efforts 1, 4, 7, 8, 10. Efforts 2/3, 5/6, high-density mode,
and the corresponding maximum-compression aliases are checked against their
representative policy on compact inputs at distance 1.2. Metal outputs must be
byte-identical within each alias group. Libjxl effort 7 is a fixed comparison policy;
equal effort numbers are not asserted to represent equal search effort.

## Acceptance contract

Every measured output must encode, independently decode to the expected finite
RGB float32 pixels, and receive a finite nonnegative Butteraugli score. Each
resident evaluation encodes twice in independent processes and requires equal
bytes on the same backend. The encoder's reported backend must be Metal fully
resident; CPU fallback and exact-coefficient execution fail this check.

All encoders receive the same canonical linear-sRGB PFM. Scoring uses pinned
libjxl Butteraugli, `RGB_D65_SRG_Rel_Lin`, and intensity target 255. GJXL's
internal AQ score is not a quality oracle. Encoders retain their own policies.

### Same-distance regression (default)

For every image, requested distance, and GJXL policy, encode Metal at that
distance. Encode libjxl effort 7 at exactly the same distance once and reuse the
reference across all eleven GJXL policies. Independent decode and Butteraugli
scoring remain mandatory for both encoders; resident repeatability and alias
checks remain unchanged. There is no calibration loop in this mode.

Reports retain both scores, both byte counts, both distances, the Metal/libjxl
byte ratio, and the signed quality difference `Metal score - libjxl score`.
Positive quality differences mean greater measured distortion in Metal output.
The byte ratio is a **same-distance size comparison**, not a matched-quality
compression-efficiency claim. A smaller file with worse quality is visible in
these separate measurements; it is not automatically a regression-test pass.
No libjxl size allowance is applied in this mode.

Without an accepted baseline, completed cases are `observed`, not `pass`. The
report records `mode: observation`, `passed: false`, and
`qualification_complete: false`. `measurement_complete` describes whether all
cases encoded, decoded, and were scored; it does not require the scores to match.

After reviewing the initial quality and size observations, a user can explicitly
accept that complete report as the Metal regression baseline:

```sh
python3 tools/metal_resident_qualification.py accept-baseline \
  --report build/release/metal-qualification/compact-same-distance-run/report.json \
  --output /path/to/reviewed-compact-baseline.json
cmake -S . -B build/release \
  -DGJXL_METAL_QUALIFICATION_BASELINE=/path/to/reviewed-compact-baseline.json
cmake --build build/release --target metal-resident-qualify-compact
```

The acceptance command validates completeness, finite measurements, same-distance
reference settings, and failure-free results before writing a new baseline. It
never overwrites a baseline or rewrites the original observation report. Failed,
incomplete, interrupted, or matched-quality observations cannot be accepted by
this command. This explicit snapshot acceptance establishes regression values;
it does not certify that the expanded resident interval is production-ready.

Subsequent runs use `--baseline` or the corresponding CMake baseline path. Each
Metal case is checked against its own approved fixed-distance values:

- Butteraugli may worsen by at most `max(0.10, 10% of baseline)`.
- Bytes may increase by at most 10%.

These are independent gates, so a quality improvement cannot hide a size
regression, and a size reduction cannot hide a quality regression. Libjxl's
same-distance measurements remain available for interpreting the results.
Reports distinguish completed regression checks (`regression_complete`) from
matched-quality qualification (`qualification_complete`).

### Matched-quality comparison (opt-in)

`--comparison matched-quality` retains the previous libjxl-only search. Keep this
on the compact suite for routine comparative analysis; explicitly requesting the
full matrix remains supported for dedicated qualification or benchmarking.

Metal stays at its requested distance while libjxl's distance varies to match
Metal's independently measured score. The search covers distances 0.03–15 with
at most 24 distinct evaluations and stops at the first candidate within 0.015
absolute **or** 2% relative Butteraugli error. Discontinuities and bounded-search
exhaustion remain `incomplete`, never a successful match. Initial matched-quality
runs without a reviewed size limit are observations.

`--max-libjxl-size-ratio`, or CMake's
`GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO`, applies **only in this mode**.
For example, 1.10 requires at most 10% size overhead at matched quality; it is not
an approved default. Same-distance runs reject this option to avoid treating
unequal-quality byte counts as a compression-efficiency gate.

With a matched-quality baseline, the fixed-distance Metal quality and size gates
above also apply, and the matched-quality Metal/libjxl byte ratio may worsen by
at most 10% while remaining within the explicit initial allowance.
`--record-baseline NEW.json` records only a passing comparison with an explicit
allowance or a passing regression against an existing accepted baseline. It does
not automatically accept initial same-distance observations.

### Baseline and cache compatibility

Report/baseline schema 3 records `comparison_mode` in the contract and each row.
Same-distance and matched-quality baselines and resume directories cannot be
mixed. Schema-1 CPU-based and schema-2 matched-quality reports remain historical
observations; they are not accepted as schema-3 baselines. CMake uses distinct
`<suite>-<comparison>-run` output directories. Compact and full baselines remain
separate through `GJXL_METAL_QUALIFICATION_BASELINE` and
`GJXL_METAL_QUALIFICATION_FULL_BASELINE`.

Evaluation cache identities are unchanged: either mode can reuse an independent
Metal or libjxl evaluation when its exact binary, input, backend, options,
distance, metric, hardware, and OS match. A changed GJXL binary triggers fresh
evaluations. Reference encodes at the same image and distance share cache entries
across GJXL policies; no policy-dependent quality is assumed equal.

## Artifacts and long runs

The complete same-distance suite covers thousands of image/policy/distance
combinations, including maximum compression. It needs at most 1,472 distinct
libjxl reference encodes for the full matrix (46 images times 32 distances),
in addition to the Metal evaluations. The opt-in matched-quality mode adds
bounded libjxl searches. Use the compact target during ordinary development. The full
target is intended for release qualification and dedicated scheduled machines.
No hosted GPU workflow or performance claim is added here.

The standalone runner accepts explicit binary, corpus, reference-manifest,
cache, baseline, and output paths; see `run --help`. `--jobs 2` processes two
independent inputs concurrently. The default is one worker to limit GPU memory.
Evaluation caches bind the binary hashes, backend, options, input, metric,
hardware, and OS. Advisory file locks protect shared caches across runs.
Scores can be reused only for identical independently decoded pixel hashes and
the same source and reference metric. Calibration PFMs are removed after
validation to avoid unbounded disk usage.

`cases/*.json` records each completed case atomically. Reusing the same output
directory resumes an identical run; changed runner/configuration identities
require a new output directory. `Ctrl-C` finishes active cases, records a partial
report, and exits 130. Missing cases and interrupted runs cannot pass. Individual
subprocesses have a 15-minute timeout.

`report.json` and `report.csv` label the comparison mode and contain all case
outcomes, both scores/byte counts, reference distances, byte ratios, signed
quality differences, and failure reasons. Exit codes are 0 for a passing
regression/qualification, 1 for failure/incomplete measurement, 2 for completed
observations without acceptance criteria, and 130 for interruption. Evaluation
directories retain
encode/decode commands and diagnostics. Failed compressed and decoded samples
are retained under a shared **2 GiB** budget by default; change
`--artifact-budget-mib` for larger storage. Cases outside the budget retain the
exact commands, input hashes, and output hashes needed to reproduce their pixels.
This bounded retention is necessary on the current workspace's limited disk.

Automatic resident selection remains `[1.0, 1.2]`. Expanding it requires a
separate review of initial distance behavior and decoded quality/size evidence
across the proposed range; merely collecting or accepting a regression baseline
does not qualify expansion. Exact-coefficient selection, device/geometry qualification, and
automatic target-size and maximum-error policies are separate and unchanged.
Until qualification passes, retain `[1.0, 1.2]` and investigate the recorded
regressions without silently widening tolerances or accepting a smaller sample.

## Historical CPU-based compact results (2026-09-05)

The original schema-1 suite compared against GJXL CPU as well as libjxl. Its
results below are preserved as historical observations, not evidence that the
new contract passes. The Release encoder at `4eda200` completed all 1,056 compact cases on Apple M4
Pro: **735 passed, 22 failed, and 299 were incomplete because CPU and/or libjxl
quality matching did not converge within the declared limits**. Individual
fixed-distance quality failures can also occur in incomplete cases. These are
qualification observations, not an accepted baseline. The portable measurements
and tool identities are retained in `initial-compact.csv` and
`initial-compact.json` under `tests/metal_qualification/`.

For example, the padded gradient at effort 7 and distance 1.0 produced resident
Butteraugli **1.2753965855**, versus CPU **1.0364149809**. Resident output was
845 bytes; matched-quality CPU output was 580 bytes at distance 1.1640625 and
Butteraugli 1.2650718689. The 45.7% size increase and quality worsening both fail
the agreed gates. Matched libjxl output was 593 bytes at Butteraugli 1.2684662342.
This example is inside the existing automatic interval; that interval is not a
universal guarantee that these newly established acceptance limits hold.

The runner refused to record this failed qualification as a baseline, and
automatic selection remains `[1.0, 1.2]`. No full-range qualification is claimed.
Portable runner tests pass, as do the focused Metal/CLI/rate-control tests and
all 22 pinned-decoder conformance fixtures. The ordinary full CTest run passed
68/69; its sole `quantization_pipeline` score mismatch was reproduced using the
original main checkout's test binary with the same actual and expected scores.

## Historical matched-quality compact observations (2026-09-05)

The previous schema-2 matched-quality runner completed all 1,056 compact cases using the same encoder
binary and compatible existing evaluation caches. **777 cases matched libjxl;
279 remain incomplete because libjxl quality matching did not converge.** No
GJXL CPU encoding, matching, or alias evaluation is requested by this suite.
No initial libjxl size limit has been approved, so the 777 matched cases are
`observed`, not `pass`; this is not an accepted baseline. Fixed-distance Metal
baseline gates remain available for subsequent qualified regression runs.

Portable observations are `tests/metal_qualification/libjxl-only-compact.csv`
and `libjxl-only-compact.json`. The original CPU-based artifacts above remain
unchanged. The previous full sweep was stopped gracefully with its partial
report intact, and a matched-quality full observation sweep reused compatible Metal/libjxl
evaluations. That sweep was subsequently stopped gracefully for the switch to
same-distance regression. Automatic resident selection remains
`[1.0, 1.2]`.

## Same-distance compact observations (2026-09-05)

The schema-3 default runner completed **all 1,056 compact cases as observations**:
all encoded, independently decoded, and produced finite scores, with no incomplete
cases. The reports retain different measured quality values at equal requested
distances; they do not assume quality equality or claim compression superiority.
The 96 distinct libjxl image/distance evaluations were shared across eleven GJXL
policies. This run reused compatible caches and is not a fresh timing benchmark.

The portable records are `tests/metal_qualification/same-distance-compact.csv`
and `same-distance-compact.json`. No Metal baseline has been accepted and the
resident selection interval remains `[1.0, 1.2]`. The full matrix has restarted
in same-distance observation mode. The optional schema-3 matched-quality compact
run was also checked: it retains 777 matched observations and 279 incomplete
matches, which do not affect the separate same-distance report.
