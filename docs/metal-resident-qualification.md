# Metal resident quality regression and rollout qualification

The runner compares independently decoded fully resident output with CPU GJXL,
pinned libjxl, and an explicitly accepted baseline. It does not require equal
encoder decisions or codestreams between implementations. A failed or incomplete
qualification never expands automatic backend selection and cannot be recorded
as an accepted baseline.

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
suite. These targets are opt-in and are not build dependencies of ordinary tests.
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
integration with CTest. Build `gjxl_pinned_quality_tools` and
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
representative policy on compact inputs at distance 1.2. CPU and Metal outputs
must agree within each alias group. Libjxl effort 7 is a fixed comparison policy;
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

For each image, policy, and requested distance:

- Resident Butteraugli must be no worse than CPU plus `max(0.10, 10% of CPU)`.
- Resident bytes must be at most 110% of CPU bytes at matched decoded quality.
- Initial libjxl size ratios are reported without a requirement that GJXL win.

CPU and libjxl matching search distances 0.03–15 with at most 24 distinct
evaluations, accepting an absolute Butteraugli difference of 0.015 or a relative
difference of 2%. A previously matched distance is a hint that must be re-encoded
and verified for the requested policy. Discrete quality gaps and bounded-search
exhaustion are **incomplete**, not successful matches. CPU and libjxl matching
are attempted independently, so one failure does not erase the other evidence.

With an accepted baseline, additional per-case checks limit fixed-distance
Butteraugli worsening to `max(0.10, 10% of baseline)`, fixed-distance byte growth
to 10%, and deterioration of the matched-quality GJXL/libjxl size ratio to 10%.
The fixed-distance checks prevent recalibration from hiding a change in GJXL's
distance behavior. Aggregate improvements cannot compensate for failing cases.

`--record-baseline NEW.json` writes a separate baseline only when every required
case passes. Existing baseline files are never overwritten. Use `--baseline`
for subsequent runs, or configure `GJXL_METAL_QUALIFICATION_BASELINE` and
`GJXL_METAL_QUALIFICATION_FULL_BASELINE` for the CMake targets. Suites have separate
baselines. Input, metric, policy, hardware, and reference-tool mismatches are
errors; a changed GJXL binary is expected and triggers fresh evaluations.

## Artifacts and long runs

The complete suite is expensive: it includes bounded CPU and libjxl searches
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

`report.json` and `report.csv` contain all case outcomes, scalar quality and
size measurements, and explicit failure reasons. Evaluation directories retain
encode/decode commands and diagnostics. Failed compressed and decoded samples
are retained under a shared **2 GiB** budget by default; change
`--artifact-budget-mib` for larger storage. Cases outside the budget retain the
exact commands, input hashes, and output hashes needed to reproduce their pixels.
This bounded retention is necessary on the current workspace's limited disk.

Automatic resident selection may expand to `[0.5, 3.0]` only after every full
case passes. Exact-coefficient selection, device/geometry qualification, and
automatic target-size and maximum-error policies are separate and unchanged.
Until qualification passes, retain `[1.0, 1.2]` and investigate the recorded
regressions without silently widening tolerances or accepting a smaller sample.

## Initial compact results (2026-09-05)

The Release encoder at `4eda200` completed all 1,056 compact cases on Apple M4
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
