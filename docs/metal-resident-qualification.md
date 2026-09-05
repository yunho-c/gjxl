# Metal resident quality regression and rollout qualification

The runner compares independently decoded fully resident GJXL output with
pinned libjxl and an explicitly accepted Metal baseline. Libjxl is the sole
reference encoder in this end-to-end suite; no GJXL CPU encodes are requested. It does not require equal
encoder decisions or codestreams between implementations. A failed or incomplete
qualification never expands automatic backend selection and cannot be recorded
as an accepted baseline. An initial run without a reviewed libjxl size limit
collects observations and cannot pass qualification.

## Run

Initialize recursive submodules and install Python 3, CMake, Ninja, and
ImageMagick. Actual qualification requires the currently qualified Apple M4 Pro.
The Python decision tests do not require a GPU or reference binaries.

```sh
python3 tests/metal_resident_qualification_test.py
just metal-resident-qualify compact
just metal-resident-qualify full
```

Both Just recipes use an isolated Release build under `build/release`. Their
CMake targets build statically linked, pinned `cjxl`, `djxl`, and
`butteraugli_main`; record their binary hashes; prepare the corpus; and run the
suite. With no baseline or size limit, these collect observations and return a
nonzero exit status. These targets are opt-in and are not build dependencies of ordinary tests.
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
integration with CTest. This requires an accepted compact baseline or an explicit
`GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO`; configuration fails otherwise. Build `gjxl_pinned_quality_tools` and
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

For each image, GJXL policy, and requested distance, encode the resident output
and find a libjxl effort-7 distance matching its independently measured quality.
Equal nominal distance and effort numbers are not treated as equivalent quality
or search policies. The runner reports Metal bytes divided by matched-quality
libjxl bytes, along with both scores and the matched libjxl distance.

Matching searches distances 0.03–15 with at most 24 distinct evaluations. It stops
immediately at the first candidate with an absolute Butteraugli difference of
0.015 **or** a relative difference of 2%, including initial guesses, endpoints,
and interior candidates. Previously successful distances are hints that must be
verified. Discrete quality gaps and bounded-search exhaustion are **incomplete**.

Without an accepted baseline or `--max-libjxl-size-ratio`, cases are **observed**,
not passed. No size threshold is inferred from the observed output. The report
records `mode: observation`, `passed: false`, and `qualification_complete: false`;
`measurement_complete` separately describes whether every case was measured and
matched. This prevents removal of the former CPU gates from silently qualifying
an encoder.

After reviewing the observations, explicitly set the maximum acceptable
Metal/libjxl byte ratio for every case. For example, `--max-libjxl-size-ratio 1.10`
requires at most 10% size overhead at matched quality. This is an example, **not
an approved default**. The equivalent CMake setting is
`GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO`. The limit is part of the contract;
changing it invalidates an existing baseline or resume directory.

An accepted Metal baseline adds independent per-case checks:

- Fixed-distance Butteraugli may worsen by at most `max(0.10, 10% of baseline)`.
- Fixed-distance bytes may increase by at most 10%.
- The matched-quality Metal/libjxl size ratio may worsen by at most 10%, while
  also remaining within the initial explicit size limit.

The fixed-distance checks prevent recalibration from hiding changes in GJXL's
distance behavior. Aggregate improvements cannot compensate for failing cases.
Initial qualification still requires reviewing whether the observed quality
versus requested distance is suitable before accepting a baseline or expanding
the resident range; a size-ratio threshold alone is not that review.

`--record-baseline NEW.json` writes a separate baseline only when an explicit
size limit exists and every required case passes. Observation-only, failed, or
incomplete runs cannot be recorded. Existing baseline files are never overwritten.
Use `--baseline` for subsequent runs; its size limit is inherited unless explicitly
supplied (and must match). CMake provides separate
`GJXL_METAL_QUALIFICATION_BASELINE` and `GJXL_METAL_QUALIFICATION_FULL_BASELINE`
paths. Inputs, metrics, policies, hardware, reference tools, and acceptance
contracts must match. Changed GJXL binaries trigger fresh evaluations.

Report/baseline schema 2 identifies the libjxl-only contract and updated
calibration stopping rule. Schema-1 CPU-based baselines and reports are not
accepted as new baselines or resumed as new cases. Existing independent Metal
and libjxl evaluation caches remain reusable when their binary/input/metric
identities match. Existing CPU-versus-Metal component tests are unchanged.

## Artifacts and long runs

The complete suite includes bounded libjxl searches
for thousands of image/policy/distance combinations, including maximum
compression. Use the compact target during ordinary development. The full
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

`report.json` and `report.csv` contain all case outcomes, Metal and matched
libjxl scores/bytes, matched libjxl distances, size ratios, and failure reasons.
Exit codes are 0 for passing qualification, 1 for failure/incomplete measurement,
2 for completed observations without a size limit, and 130 for interruption. Evaluation directories retain
encode/decode commands and diagnostics. Failed compressed and decoded samples
are retained under a shared **2 GiB** budget by default; change
`--artifact-budget-mib` for larger storage. Cases outside the budget retain the
exact commands, input hashes, and output hashes needed to reproduce their pixels.
This bounded retention is necessary on the current workspace's limited disk.

Automatic resident selection may expand to `[0.5, 3.0]` only after every full
case passes the reviewed libjxl-only contract and the initial distance behavior
has been reviewed. Exact-coefficient selection, device/geometry qualification, and
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

## Libjxl-only compact observations (2026-09-05)

The schema-2 runner completed all 1,056 compact cases using the same encoder
binary and compatible existing evaluation caches. **777 cases matched libjxl;
279 remain incomplete because libjxl quality matching did not converge.** No
GJXL CPU encoding, matching, or alias evaluation is requested by this suite.
No initial libjxl size limit has been approved, so the 777 matched cases are
`observed`, not `pass`; this is not an accepted baseline. Fixed-distance Metal
baseline gates remain available for subsequent qualified regression runs.

Portable observations are `tests/metal_qualification/libjxl-only-compact.csv`
and `libjxl-only-compact.json`. The original CPU-based artifacts above remain
unchanged. The previous full sweep was stopped gracefully with its partial
report intact, and a new full observation sweep reuses compatible Metal/libjxl
evaluations under the new contract. Automatic resident selection remains
`[1.0, 1.2]`.
