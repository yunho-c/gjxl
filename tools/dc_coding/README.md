# Current-effort DC prediction qualification

This harness qualifies the explicit weighted predictor against gradient in a
fresh native build. Collection and reporting are separate commands. It does
not launch calibration or change encoder policy.

## Builds

The candidate is a fresh Release build with tests and benchmarks enabled. The
baseline is a separate fresh Release build from clean `2c936fa9`, with benchmarks
enabled. `init` rejects pending builds and fingerprints the source, shaders,
compiler commands, CMake caches, executable files, and decoder libraries.
Sources are retained in `source.zip`; the tracked diff is retained separately.

The decoder helper links only to an explicitly supplied pinned libjxl build,
outside the native encoder. On macOS:

```sh
python3 tools/dc_coding/build_decoder.py \
  --source third_party/libjxl \
  --build /path/to/pinned-libjxl-build \
  --output build/dc-tools

cmake --build build/release --target \
  gjxl_quality_benchmark gjxl_dc_prediction_benchmark
```

The pinned source revision is `e8ff09762481785938d8e4e01333ed3917571161`.
`build.json` records the helper compilation and decoder dynamic-library hashes.

## Collection

The current study inputs are selected deterministically from the saved runtime
study metadata and historical native-weighted report: all 32 CLIC photographs,
all 24 Kodak photographs, four other 4K photographs, and eight compact controls.
Efforts 3, 4, and 7 at distances 0.6, 1, and 2 give 612 cases. These distances
are repeated observations of each image, not independent images. No threshold
is fitted to their results.

```sh
python3 tools/dc_coding/study.py init \
  --output build/dc-weighted-qualification-v1 \
  --metadata /path/to/fixed-gjxl-full-20260912/metadata.json \
  --historical /path/to/gjxl-writer-rate-attribution/docs/native-weighted-dc/summary.json \
  --baseline-source /path/to/gjxl-runtime-main-20260912

python3 tools/dc_coding/study.py collect --output build/dc-weighted-qualification-v1
python3 tools/dc_coding/study.py timing --output build/dc-weighted-qualification-v1
python3 tools/dc_coding/study.py report --output build/dc-weighted-qualification-v1
```

`--pilot` on `init` selects four inputs at distance 1 (12 cases). Optional
`--max-cases N` bounds each collection/timing invocation without changing the
manifest or claiming the full matrix is complete. Resume by repeating the same
command. A changed source/tool/input identity requires a new output directory.
Completed records are written atomically after checks pass; failed attempts are
retained. The report includes missing case IDs and timing completion counts.

Each size case retains three outputs: the untouched baseline, candidate
gradient, and candidate weighted. Baseline and candidate gradient must match
byte-for-byte. Weighted and gradient must decode to byte-identical finite
linear-sRGB float arrays. Every variant repeats during its encode invocation;
the encoder checks stable output after each call. Saved JXL files and their
hashes remain available for later audit. Pixel hashes cover native-endian float
bytes in top-to-bottom, interleaved RGB order; they do not include a PFM header.

## Timing boundaries

The separate timing phase uses a preselected subset at distance 1, all three
efforts, and 20 AB/BA pairs (five in the pilot). Encoder and decoder work run
sequentially, with process/CPU/power snapshots and rejection of concurrent
encoder, build, or sustained heavy CPU work. Ordinary OS background activity
can remain; small timing differences should be interpreted with paired samples
and sign consistency, rather than standalone medians alone.

- Encode: one process, shared warmed production Metal backend, loaded planar
  linear RGB, one validation call and three warmups per variant. Each sample
  surrounds the complete uninstrumented public call; eight CPU participants,
  fully resident AQ, default density and compression, final diagnostic scoring
  disabled. Prediction choices alternate within pairs and reverse every pair.
  Validation and file writes are outside the timer. All timed output bytes must
  match the corresponding size-case reference.
- Decode: one process with both JXL buffers loaded; each call creates a fresh
  single-thread decoder, allocates float output, decodes, and destroys the
  decoder. Equality checks and hashes are outside the timer. Output color is
  explicitly `RGB_D65_SRG_Rel_Lin`; every sample must match reference pixels.

`size.csv`, `size-summary.csv`, `timing.csv`, and `summary.json` are generated
only from retained evidence. Pilot v1's cross-process encode measurements are
diagnostic and are superseded by the paired-call method in pilot v2. Weighted
prediction preserves quality, so its byte savings need no quality recalibration.
DC quantization/smoothing experiments change reconstruction and use their own
independent measured-quality curves below.

## Lossy DC processing study

`processing_study.py` is a separate namespace for reconstruction-changing
experiments. Its five controls are the unchanged gradient default, weighted
residual coding, weighted prediction-aware quantization, weighted smoothing,
and weighted quantization plus smoothing. The weighted control isolates the
lossy changes; the gradient control reports their combined effect against the
current default. Gradient prediction-aware arithmetic is also covered by the
CPU, device, workflow, and independent-decoder tests.

The full matrix deterministically selects six CLIC and six Kodak photographs
spread across their sorted manifests, two 4K photographs, and four compact
controls. Efforts 3/4/7 at distances 0.5/0.7/1/1.4/2 give 270 curve cases and
1,350 retained codestreams. The pilot uses one CLIC, one Kodak, edge, and
gradient at three distances. Collection repeats each native encode and checks
unchanged default artifacts wherever the previous qualification overlaps.
Weighted and gradient baseline decoded float arrays must be exactly equal.

```sh
cmake --build build/release --target gjxl_dc_processing_benchmark
python3 tools/dc_coding/build_decoder.py \
  --source third_party/libjxl --build build/dc-libjxl-reference \
  --output build/dc-processing-tools-NEW
python3 tools/dc_coding/processing_study.py init --output build/dc-processing-NEW --pilot \
  --decoder-build build/dc-processing-tools-NEW/build.json
python3 tools/dc_coding/processing_study.py collect --output build/dc-processing-NEW
python3 tools/dc_coding/processing_study.py calibrate --output build/dc-processing-NEW
python3 tools/dc_coding/processing_study.py timing --output build/dc-processing-NEW
python3 tools/dc_coding/processing_study.py report --output build/dc-processing-NEW
```

Replace `NEW` with a fresh run name. Omit `--pilot` for the full matrix. Existing
qualification and pilot namespaces bind their original sources and artifacts;
do not overwrite their tools or resume them against changed sources.

`init` freezes source, shaders, build commands, native binaries, decoder and
scorer libraries/versions, input hashes, and the matrix. Collection commands
reject changed identities and concurrent collector processes. `--max-cases N`
bounds one invocation. `init --equivalent-study PREVIOUS_DIR` also binds any
completed overlapping cases to their previous codestream hashes, for exact
algorithm-preserving implementation changes. Reporting only reads saved artifacts, lists incomplete
cases, and never starts a codec. Generated float PFMs are removed only after
successful scoring; JXL files, decoder commands, pixel hashes, scorer output,
and failed attempts are retained. Scores use independently decoded linear-sRGB
float pixels and the pinned fast-ssim2 adapter from the runtime notebook study.

BD-rate integrates PCHIP log(bytes) against measured SSIMULACRA2. Each image /
effort uses the common quality interval of all five controls, without
extrapolation. Strictly dominated points are removed from each interpolation
curve; all observations remain in `curves.csv`. Curves with fewer than three
frontier points or no common quality interval are explicitly excluded. Compact
controls are reported separately from photographs.

Timing uses the distance-1 default's measured quality. Each lossy variant is
calibrated within 0.05 SSIMULACRA2 using retained bracketed probes (at most ten
new probes per variant). Unbracketed/exhausted cases remain explicitly
unresolved, with no matched-quality timing claim. The selected timing subset
contains one CLIC, one Kodak, one 4K photograph, and the edge control at e3/e4/e7
(the pilot omits 4K). Each process loads the source once, validates every
variant, performs three warmups, and measures 20 rotated/reversed rounds of
complete public encode calls sharing one production Metal backend. Distances
may differ by variant; timed bytes must match calibrated artifacts.

The native benchmark also accepts `cpu` for diagnostic checks. Its variant
specification has one line per control:

```text
default 1 gradient round 0
weighted 1 weighted round 0
quantize 0.95 weighted prediction-aware 0
smooth 1.05 weighted round 1
both 1 weighted prediction-aware 1
```

Invocation: `gjxl_dc_processing_benchmark input.pfm new-output-dir variants.txt
effort threads warmups rounds cpu|metal`. The last three columns select
predictor, quantizer, and smoothing; supplied distances must be independently
measured before calling a comparison matched-quality.

The decoder helper's optional `--independent-pixels` permits lossy comparisons
while still checking finite, deterministic pixels independently for each
stream. Its default mode continues to require equality between streams. Decode
cost includes creation, output allocation, decode, and decoder destruction,
with one CPU thread, three warmups, and alternating pairs. Encode and decode
run sequentially, without scorer or build work running alongside them.

The completed bounded qualification is `build/dc-processing-qualification-v1`:
270/270 curve cases and 10/12 calibrated timing cases. Edge effort 3 (combined)
and effort 7 (quantization) exhausted the declared calibration budgets and
remain explicitly excluded from matched-quality timing. The
[portable report](../../docs/dc-processing-qualification/REPORT.md) includes
all results, negative findings, and the unchanged-default decision.
`docs/dc-processing-qualification/analyze.py` reproduces its CSVs, artifact audit,
paired timing summaries, and linear-interpolation sensitivity from saved data
without running a codec or scorer.
