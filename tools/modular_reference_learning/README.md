# Reference Modular learner bridge

Research and qualification tools for the
[learner replacement plan](../../docs/modular-tree-learning-cost.md#recommended-implementation-order).
The native one-fit histogram workflow is now the public learned CPU policy;
older archives below describe their recorded revisions, not current defaults.
The reference is libjxl
`e8ff09762481785938d8e4e01333ed3917571161`; qualified native archives come from
`f27a797` (`perf/modular-ans-estimates`).

## CPU-hybrid histogram qualification

The root-CMake target `gjxl_modular_hybrid_profile` requires tests enabled and
`GJXL_MODULAR_ORACLE_BUILD` pointing to the pinned libjxl build. Enable Metal
for hybrid routes. The driver encodes original packed pixels with the same
native histogram workflow, independently decodes the result, checks resource
release, and writes `output.jxl` and `result.json` to a fresh directory.

```sh
cmake --build build/release --target gjxl_modular_hybrid_profile
build/release/gjxl_modular_hybrid_profile INPUT.pam OUT auto auto best
python3 tools/modular_reference_learning/run_hybrid_qualification.py \
  --binary build/release/gjxl_modular_hybrid_profile \
  --input INPUT.pam --out build/histogram-hybrid-qualification \
  --routes cpu auto --warmups 1 --repeats 3
```

Driver arguments are `INPUT OUT ROUTE BUFFERING PREDICTORS`:

- `cpu`: CPU workflow, with no Metal backend; the private driver defaults to
  serial stages unless explicit participant limits are supplied.
- `shared-cpu`: bounded CPU feature preparation shared by learning and Metal
  token generation.
- `hybrid`: requests GPU features for supported shapes, plus Metal tokens;
  oversized metadata or unavailable optional feature pipelines select CPU
  preparation before execution.
- `auto`: chooses the feature producer using the existing batch crossover,
  plus Metal tokens. This route name does not mean public automatic backend
  selection; with the trailing `public` flag it selects explicit Metal.

Buffering is `auto|buffered`; predictors are `best|all`. The repeated runner
holds these at `auto best`, uses the same four-participant CPU ceiling and
estimated direct ANS/64-cluster policy for every route, and rotates serial
fresh-process route order. One warmup per route/input precedes measured calls.
The private driver's default CPU control remains serial. Use repeated `--input` arguments for
multiple images. With no `--routes`, all four controls run; check the projected
runtime before launching a large corpus.

Commit implementation changes before running the harness: it requires a clean
worktree, freezes the executable, records revision/build/dependency/input
hashes and start/end power conditions, saves raw results and output hashes,
and rejects changed bytes or learning-work counts across routes/repetitions.
Each run needs a fresh output directory. The complete API timer includes
selection, feature/sample preparation, fitting, tokens, entropy, admission,
output copy and prepared-frame teardown. It excludes input/output I/O, backend
initialization, reference source unpacking and independent decoding. Feature,
sample-preparation and fitting details are nested within training; GPU token
details are nested within tokens. Managed memory excludes caller input and
reference-decode buffers, stack, allocator bookkeeping and backend initialization.
Use the [integration plan](../../docs/gpu-histogram-learner-integration.md)
for ownership, admission, routing and validation boundaries.
The [2026-10-14 qualification](../../docs/gpu-histogram-learner-qualification.md)
records matched full-resolution and crossover results on Apple M4 Pro.

## Integrated public CPU/Metal profiling

The [integrated profile](../../docs/modular-parallel-workflow-profile.md) compares
the public learned encoder with explicit CPU and Metal backends, a fixed
14-participant domain, and the integrated parallel stages. It includes complete
call timing, local learning-wave/job lifetimes, detailed fitting phases and
separate all-thread Samply captures. Job lifetimes measure scheduling activity,
not hardware CPU utilization.

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam OUT auto auto best schedule 14 14 14 14 14 public
python3 tools/modular_reference_learning/run_parallel_profile.py \
  --binary build/release/gjxl_modular_hybrid_profile \
  --input /path/to/photo-large-1029.pam \
  --input /path/to/photo-large-1039.pam \
  --input /path/to/photo-large-1049.pam \
  --out build/parallel-workflow-profile --repeats 3
```

The trailing `public` flag selects schema 6 and requires `auto best` policy
arguments with matching learning, CPU-ceiling, entropy, selection and token
participant limits. `fit-off` keeps top-level timers; `schedule` adds wave/job
records; `fit-on` adds detailed fitter probes as well. Diagnostic storage and
backend initialization are outside the encoder timer. The runner freezes local
source edits, checks AC power and historical output hashes, and retains all
observations. See the report for analysis commands and measurement boundaries.

The [fitting scheduling follow-up](../../docs/modular-fitting-scheduling.md)
uses `run_fitting_schedule_qualification.py` to compare the frozen profiler
from that capture with a newly built executable. It interleaves CPU/Metal
before/after controls and retains separate schedule traces. Local feature
cohorts can now span several producer batches under the same aggregate sample
ceiling; `learning.waves` counts joined queue invocations, not groups of one
job per participant.

The [work-reduction integration](../../docs/modular-fitting-work-reduction-integration.md)
uses `--baseline-variant after` to select that qualified scheduling executable
as its control. `--diagnostic-mode fit-on` adds separate detailed captures and
checks logical fitter-work parity across implementations, with complete work
counter equality across CPU/Metal routes within each implementation.

The [GPU histogram fitter contract](../../docs/gpu-histogram-fitting.md) starts
from the consolidated CPU scheduler, shared-fit property parallelism, ANS
allocation reuse and training-row hash reuse. Its
[paired baseline](../../docs/gpu-histogram-fitting-baseline.md) uses the same
runner and `fit-on` diagnostics. The contract adds planning and validation only;
the public encoder still fits trees on CPU.

## Parallel local learning

Append explicit learning participants and a fixed CPU ceiling after the fitting
profile mode to measure local CPU preparation/fitting concurrency, or property
evaluation within a shared-tree fit:

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam OUT auto auto best fit-off 14 14
python3 tools/modular_reference_learning/run_parallel_learning.py \
  --binary build/release/gjxl_modular_hybrid_profile \
  --input INPUT.pam --out build/parallel-learning-qualification \
  --reference docs/gpu-histogram-learner-integration-results.json \
  --participants 1 4 14 --cpu-ceiling 14
```

The runner freezes the executable and tracked/untracked build sources, rotates
participant counts, requires AC power on macOS, and checks complete bytes
against the earlier qualification. Supply `--input` once per photograph. Each
configuration has one warmup and three measured repetitions by default.

Explicit participant arguments select schema 2: `detail_wall.learning` is
elapsed preparation/fitting time; `detail_work` contains overlapping per-job
time sums. Cohort alignment can change feature batch boundaries and automatic
producer routing across participant counts; the runner records those changes
and requires stable routing within each repeated condition. Older invocations keep
serial learning and schema 1. See the [local fitting implementation](../../docs/parallel-tree-fitting.md)
and [shared-tree property parallelism](../../docs/parallel-shared-tree-fitting.md).

## Histogram fitting attribution

The same driver accepts an optional final `fit-off|fit-on` argument (default
`fit-off`). The private histogram profile has allocation-free counters and
task wall phases inside `LearnHistogramTree`, plus inclusive node times
by distinct-row count/depth, bin/symbol-span distributions and per-fit records.
It does not reuse the original sort-based learner's attribution. Candidate
evaluation combines prefix histogram updates, entropy costs and predictor
comparison; no clock runs per threshold. Diagnostic arrays are caller-owned,
allocated outside the encoder timer, and excluded from codec admission.

Histogram profile schema 3 adds `peak_property_participants` and
`parallel_property_nodes`. Fit/node durations remain elapsed spans, while
property phase durations sum task wall intervals and can overlap within a fit.
The exclusive phase attribution report requires serial property evaluation;
it rejects parallel captures rather than subtracting overlapping work from
elapsed time. Readers still accept schemas 1 and 2.

Histogram profile schema 2 introduced fused first-predictor occupancy in
`joint_histograms`. Physical property/histogram row passes are
`property_row_visits + joint_histogram_row_visits - fused_property_row_visits`;
`skipped_constant_properties` counts inherited/one-bin constants needing no row
scan. Logical property/bin/candidate counters retain their previous meanings.
The reader also accepts historical schema 1. See the
[implementation and matched measurements](../../docs/modular-fitter-work-reduction.md).

```sh
python3 tools/modular_reference_learning/run_histogram_attribution.py \
  --binary build/release/gjxl_modular_hybrid_profile \
  --input /path/to/photo-large-1029.pam \
  --input /path/to/photo-large-1039.pam \
  --input /path/to/photo-large-1049.pam \
  --reference docs/gpu-histogram-learner-integration-results.json \
  --out build/histogram-fitting-attribution --warmups 1 --repeats 3
python3 tools/modular_reference_learning/analyze_histogram_attribution.py \
  build/histogram-fitting-attribution \
  --json docs/modular-histogram-fitting-profile-results.json \
  --markdown docs/modular-histogram-fitting-profile.md
```

This runner compares attribution on/off through the same automatic hybrid
policy and binary. It checks exact decoding, bytes, routing and work counters,
freezes the executable, archives/hashes all build/tool sources including local
edits, and rejects source/input/dependency changes during capture. Unlike the
older route-qualification runner, it records a source snapshot rather than
requiring a commit. Its base revision alone does not identify that snapshot.
Before/after power, load and thermal observations accompany every call.

The analyzer rejects an image cohort whose observed power source changes.
Repeat that entire input cohort under stable power, then pass the completed
capture with `--replacement build/replacement-capture`. The replacement must
use the identical frozen binary and input. All original calls remain in the
JSON; only the complete replacement cohort supplies timing medians. Reports
label power per image and keep node-time shares separate across images.

For a separate sampling diagnostic, build with `-O3 -g -DNDEBUG`, retain a
matching `dsymutil` bundle, and record the driver directly with
`samply record --save-only --main-thread-only --unstable-presymbolicate` using
`fit-off`. `analyze_fitting_samply.py PROFILE.json.gz --binary BINARY --output
ANALYSIS.json` resolves the native fitter and its inline source frames using
`llvm-symbolizer`. It checks symbol coverage and reports both current-stack
and preceding-stack assignments of sampled CPU deltas. These CPU samples are
not additive wall-clock phases and the sampled run is excluded from paired
timing medians.

## Conditional native WP search

The optional root-CMake target `gjxl_modular_reference_wp_sweep` accepts
`INPUT OUT auto|buffered best|all`. It uses original packed pixels and the
native reference-style workflow, with libjxl only for independent decoding.
It checks a fresh replay of the selected WP vector, then refits/codes each of
five uniform WP assignments with transforms and tree scope fixed. These are
conditional controls, not exhaustive per-stream transform search or a runtime
selector. Existing output directories are rejected.

```sh
cmake --build build/reference-wp-sweep --target gjxl_modular_reference_wp_sweep
ctest --test-dir build/reference-wp-sweep -R '^modular_reference_wp_sweep$' --output-on-failure
python3 tools/modular_reference_learning/run_reference_wp_sweep.py \
  --binary build/reference-wp-sweep/gjxl_modular_reference_wp_sweep \
  --validation build/reference-workflow-validation \
  --out build/reference-wp-sweep-study
python3 tools/modular_reference_learning/analyze_reference_wp_sweep.py \
  build/reference-wp-sweep-study \
  --validation build/reference-wp-sweep-validation-v2 \
  --out /tmp/gjxl-reference-wp-sweep.json
```

Configure a fresh native build with tests enabled and
`GJXL_MODULAR_ORACLE_BUILD` pointing at the pinned libjxl build, as for the
other native oracle targets. Use fresh study/output paths. The runner freezes
its executable, source, build inputs, original pixels, qualified files and
results. The analyzer's optional validation argument binds the archived
sanitizer/file-comparison/rejection-test supplement. Without that local
supplement, omit the argument; it does not run new encodes. See
[the WP study](../../docs/modular-reference-wp-sweep.md) for the measured
opportunity, domain limitations and remaining gates.

## Reference policy and scope controls

`reference-policy INPUT OUT trace|timing auto|buffered` changes only the input
buffering setting when `buffered` is requested; automatic mode preserves the
stock e9 API sequence. Generate/build in fresh `policy-generated` / `policy-bin`
directories using the build recipe below with those paths substituted. Keep
older generated sources intact: existing ledgers bind their hashes.

```sh
python3 tools/modular_reference_learning/run_policy_study.py \
  --binary build/reference-learning/policy-bin/reference_learning \
  --corpus build/reference-learning/corpus \
  --output build/reference-learning/reference-policy \
  --inputs photo-macan photo-flower alpha-flower chessboard splines --repeats 3
python3 tools/modular_reference_learning/test_policy_trace.py \
  --case build/reference-learning/reference-policy/photo-macan-auto--1
python3 tools/modular_reference_learning/analyze_policy_study.py \
  --study build/reference-learning/reference-policy \
  --validation build/reference-learning/reference-policy-validation \
  --output docs/modular-reference-policy-results.json
```

Use a new output directory; the runner does not overwrite or silently resume a
partial study. It freezes the binary, exact sources, generated hooks and build
configuration, and binds inputs and archives by hash. Omit `--validation` if the
archived sanitizer controls are unavailable. The negative tests above use the
nonempty photographic global-tree trace.

Trace mode writes `streams/learning-N.json` and `.i32le` payloads for every
successful learning call. Included channels have tight signed 32-bit rows;
excluded channels retain descriptors but no pixel bytes. Metadata preserves
stream-ID holes, every learning option, float bits, WP coefficients, transforms,
multipliers and the raw tree. Payloads are read back and checked sample-for-sample.
`coding-N.json` records global/local coding scope and header settings separately.
Trace overhead is explicitly flagged and excluded from timing analysis. Local
scope is captured at the call site, not inferred from the number of images.
See the [ablation report](../../docs/modular-reference-policy-ablation.md) for
results and the distinctions between workflow, learning and fitting comparisons.

## Native sample preparation

`prepare-check INPUT POLICY FIXTURE best|all [OUTPUT]` borrows native transformed
stream slices, prepares compact samples, checks every captured sample field,
fits the native tree and checks every raw node field. An optional output path
also encodes through native estimated ANS and verifies an independent exact
decode. The corpus runner checks complete byte identity with the frozen bridge:

```sh
python3 tools/modular_reference_learning/run_native_preparation.py \
  --binary build/reference-learning/bin/reference_learning \
  --corpus build/reference-learning/corpus \
  --output build/reference-learning/native-preparation
python3 tools/modular_reference_learning/analyze_native_preparation.py \
  --study build/reference-learning/native-preparation \
  --corpus build/reference-learning/corpus \
  --validation build/reference-learning/native-preparation-validation \
  --output docs/modular-native-sample-preparation-results.json
```

Use a new output directory. `--inputs` restricts a diagnostic run; omit
`--validation` when the archived final build validation is unavailable.
`preparation-self-test` runs 119 generated reference/native comparisons plus
the explicit pinned-reference weight-overflow rejection regression.
`native_preparation_test` checks the native admission, atomic publication and
managed allocation/storage contract. Both and the ledger tests run under CTest.
See the [preparation report](../../docs/modular-native-sample-preparation.md) for
profile limits and timing caveats. Defaults and candidate selection are unchanged.

## Actual e9 stream preparation replay

`stream-prepare-check DESCRIPTION PIXELS` reconstructs one captured learning
call, checks the reference tree against its original capture, compares every
native prepared field, and checks the native raw tree and component storage
bounds. It preserves per-stream WP presets, excluded channel descriptors and
group-ID holes. It is a correctness check, not a timing or coding experiment.

```sh
python3 tools/modular_reference_learning/run_stream_replay.py \
  --binary build/reference-learning/stream-replay-bin/reference_learning \
  --policy-study build/reference-learning/reference-policy \
  --policy-validation build/reference-learning/reference-policy-validation \
  --output build/reference-learning/stream-replay
python3 tools/modular_reference_learning/analyze_stream_replay.py \
  --study build/reference-learning/stream-replay \
  --output build/reference-learning/stream-replay-summary.json
```

Use fresh build/output paths. The runner requires the qualified policy captures
and their validation archive, adding its true 16-bit control to the five-input
auto/buffered matrix. Optional `--cases INPUT:VARIANT ...` selects a diagnostic
subset. `stream_replay.py --trace JSON --output DESCRIPTION` projects a single
capture for direct replay; pixel payloads stay in their original bound files.
Unsupported mixed sampling/growth options are rejected, not approximated.

The combined report is reproduced with the analyzer's `--scalar`, `--sanitizer`,
`--default-wp`, `--scalar-default-wp`, `--corpus` and `--validation` arguments.
They bind full scalar replay, the three sanitizer input controls, both sets of
eight default-WP encodes, final build/test artifacts and the scalar rounding
regression. See [the qualification](../../docs/modular-native-stream-preparation.md).
Historical preparation analyzers retain their original 101-case expectation;
their old validation archives are not rewritten.

## Paired complete learning

`paired-learning INPUT POLICY FIXTURE best|all REPEATS` times preparation plus
fitting on identical immutable transformed planes and strides. Both paths warm
once, then alternate order; samples are checked in the native warmup and every
measured tree is checked after timing. Native admission/validation, allocation
and temporary sample release are included; image conversion and comparisons
are excluded. No encodes run in this mode.

```sh
python3 tools/modular_reference_learning/run_native_preparation.py \
  --binary build/reference-learning/bin/reference_learning \
  --corpus build/reference-learning/corpus \
  --output build/reference-learning/native-learning-pilot \
  --mode paired-learning --repeats 3 \
  --inputs photo-macan alpha-ria gray-patches
python3 tools/modular_reference_learning/analyze_paired_learning.py \
  --study build/reference-learning/native-learning-pilot \
  --corpus build/reference-learning/corpus \
  --validation build/reference-learning/native-learning-validation \
  --output docs/modular-native-learning-timing-results.json
```

Use a new output directory; optional `--variants best all` restricts predictor
controls. The validation argument binds archived Release/ASan test logs and
representative paired controls; omit it without that local archive. The
analyzer checks fixture/policy/input hashes against the frozen bridge, exact
checks, run order, expected shapes, raw stdout and timing nesting. Reference
non-fit time is a per-run difference, including sample release and wrapper
overhead, not an isolated preparation timer. See the
[timing report](../../docs/modular-native-learning-timing.md) for the limits.

## Native fitting replay

The harness also builds the dependency-free native fitter under ordinary
production compiler settings, separately from reference-only compiler flags.
`native_fit_test` checks the private native contract; `native-self-test` generates
129 independent differential fixtures. Both are registered with CTest.

```sh
ctest --test-dir build/reference-learning/bin --output-on-failure
python3 tools/modular_reference_learning/run_native_replay.py \
  --binary build/reference-learning/bin/reference_learning \
  --corpus build/reference-learning/corpus \
  --output build/reference-learning/native-paired \
  --mode paired --repeats 3
python3 tools/modular_reference_learning/analyze_native_replay.py \
  --study build/reference-learning/native-paired \
  --validation build/reference-learning/native-validation \
  --output docs/modular-native-histogram-learning-results.json
```

Use a new output directory; the runner refuses to overwrite any study. The
runner freezes the executable and exact source snapshots, including uncommitted
files, and binds inputs, archives and compiler settings by SHA-256. Optional
`--inputs` and `--variants` restrict a diagnostic study. `--mode verify` runs a
single native fit per fixture without warmup, for correctness rather than timing.
`--validation` additionally checks the archived final-build validation used by
the checked-in report; omit it when that local archive is unavailable. The
archive includes contract-test logs, scalar/sanitizer replay results and the
timed/final native-object equivalence check. Analyzer rejection tests are also
registered with CTest.

Direct modes are `native-verify FIXTURE`, `native-replay FIXTURE REPEATS`, and
`paired-replay FIXTURE REPEATS`. Copies/conversion are outside both timers;
native validation/allocations are inside. Every fit compares all raw tree fields.
The paired mode alternates execution order after one warmup each and measures
native storage separately during its warmup. See the
[native fitting contract](../../docs/modular-native-histogram-learning.md).

`validation-replay FIXTURE REPEATS` separately times repeated admission checks
on already converted samples. The validation-scan follow-up retains full
validation inside ordinary native fitting timers. Its combined report can be
recreated with:

```sh
python3 tools/modular_reference_learning/analyze_validation_scan.py \
  --paired build/reference-learning/native-validation-scan-paired \
  --verify build/reference-learning/native-validation-scan-verify \
  --scan build/reference-learning/validation-scan \
  --validation build/reference-learning/validation-scan-validation \
  --baseline docs/modular-native-histogram-learning-results.json \
  --output docs/modular-histogram-validation-scan-results.json
```

For scalar arithmetic qualification, configure a separate build with
`-DCMAKE_CXX_FLAGS=-DGJXL_HISTOGRAM_SCALAR=1`. This validates the scalar fallback
against the pinned four-lane reference; it does not qualify another reference
architecture or compiler. Existing ASan/UBSan builds also instrument the native
fitter and its contract/differential tests.

The original bridge analyzer resolves changed tracked tool sources through
their recorded capture commit. Frozen binaries, fixtures, archives and other
data must still match their live artifact hashes; they cannot resolve through
Git history. This preserves verification of older studies after tool evolution.

`prepare.py` checks the reference revision and pristine hooked files, records
their hashes, and generates these changes outside the reference checkout:

- Rename the original `ComputeBestTree` entry point so a capture wrapper can
  call its unchanged implementation.
- Add a layout-neutral friend declaration for sample serialization.
- Time the enclosing `LearnTree` call.
- Set the existing reference codec adapter to effort 9 instead of effort 1.

The little-endian versioned fixtures contain all quantized sample columns,
weights, token/extra-bit pairs, property and predictor order, dequantization
cuts and quantization maps, plus the exact float parameter bits, static ranges,
multiplier ranges and expected tree. They are captured before the fitter
partitions samples in place. Each capture is read back from disk and replayed
to the exact tree. The disposable dedup table must already be finalized.
Capture/replay overhead is excluded from the fit timer and explicitly included
in enclosing capture-run timers. Use `timing` runs for preparation-plus-fit
comparisons. The process peak RSS includes the entire harness, encoding and
validation; it is **not** an isolated learner scratch measurement.

The bridge copies GJXL's canonical stream slices into reference images, retaining
stream-ID holes, local channel numbering, metadata, shifted channel extents and
predictor reset boundaries. It uses the reference lossless e9 learning settings
with either Best (Weighted/Gradient) or Variable (all 14) predictors. Custom WP
headers and previous-channel properties are currently outside this adapter's
profile. No sample/leaf cap or repeated threshold search is imposed. Fixtures
have an explicit research admission limit of 16M samples and 512 MiB; captures
above native's current 1M sample ceiling are flagged, not truncated.

Imported trees must pass native property, predictor, offset, multiplier,
shape/depth and image-dependent size validation. For each bridge encode, tree
tokens and **every** residual/context token must match reference tokenization
on the same stream images. GJXL then encodes the imported tree through its
unchanged estimated-ANS path; reference decoding must reconstruct every source
integer, including hidden RGB. Timing/capture repeats must retain identical
compressed bytes. Saved stock e9 controls are compared by SHA-256 when available.

## Build

```sh
python3 tools/modular_reference_learning/prepare.py \
  --reference /Users/yunhocho/GitHub/gjxl/third_party/libjxl \
  --output build/reference-learning/generated
cmake -S tools/modular_reference_learning -B build/reference-learning/bin \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGJXL_BUILD=/Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  -DREFERENCE_SOURCE=/Users/yunhocho/GitHub/gjxl/third_party/libjxl \
  -DREFERENCE_BUILD=/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl \
  -DGENERATED_SOURCE="$PWD/build/reference-learning/generated"
cmake --build build/reference-learning/bin -j4
```

## Run

Run one study process at a time. This example captures the three pilot inputs
and interleaves one warmup and two measured fresh processes for each variant.
Omit `--inputs` for the complete saved 16-input corpus, or omit
`--timing-repeats` for capture/rate qualification only.

```sh
python3 tools/modular_reference_learning/run_study.py \
  --binary build/reference-learning/bin/reference_learning \
  --generated build/reference-learning/generated \
  --ans-study /Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  --reference-study /Users/yunhocho/GitHub/gjxl-modular-libjxl-study/build/modular-libjxl-controls \
  --output build/reference-learning/pilot \
  --inputs photo-macan alpha-ria gray-patches \
  --variants reference best all --timing-repeats 2
python3 tools/modular_reference_learning/test_capture.py \
  --binary build/reference-learning/bin/reference_learning \
  --fixture build/reference-learning/pilot/gray-patches-reference--1/fixtures/fit-0.bin
```

The manifest binds inputs, source files, binary, native/reference archives and
prior ledgers/policies. Completed cases are verified before resuming. A changed
manifest needs a new output directory; a partial case requires inspection.
Replay alone uses `reference_learning replay FIXTURE REPEATS`; it copies the
immutable captured samples outside each fitting timer and verifies every tree.

The [qualified results](../../docs/modular-reference-learning.md) cover 48
corpus encodes, 233 fitting captures/replays and 45 repeated pilot encodes.
The all-predictor bridge reaches +0.9487% aggregate bytes versus e9 on frozen
native transforms, but neither predictor policy is ready for promotion.

For boundary validation, configure a second build with the same path arguments
and `-DCMAKE_BUILD_TYPE=RelWithDebInfo` plus
`-DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-sanitize=vptr -fno-omit-frame-pointer'`.
Use `UBSAN_OPTIONS=halt_on_error=1` to run `test_capture.py`, a photographic
fixture replay, and an `alpha16-ramp` bridge encode with all predictors. The
vptr exclusion is required by the pinned no-RTTI archive; the linked archives
are uninstrumented. Preserve the sanitizer executable, CMakeCache.txt, logs
(`capture-tests.log`, `photo-replay.log`, `alpha16.log`) and output (`alpha16.jxl`)
in a separate validation directory so later builds cannot overwrite evidence.

The analyzer verifies hashes and completeness before producing the report:

```sh
python3 tools/modular_reference_learning/analyze.py \
  --corpus build/reference-learning/corpus \
  --timings build/reference-learning/pilot-qualified \
  --sanitizer-validation build/reference-learning/sanitizer-validation \
  --output docs/modular-reference-learning-results.json
```

Captured sample payloads are ignored research artifacts. The study copies its
executable into the output directory so rebuilding the harness does not replace
the binary bound to completed captures. The preliminary `pilot` was made before
final compiler/error-path qualification; use `corpus` and `pilot-qualified` for
the committed measurements.

## Same-token entropy controls

After the prepared-stream coding replay, `run_entropy_replay.py` keeps the
captured transformed samples, trees, WP headers and stream scopes fixed while
varying downstream entropy coding. The [ablation report](../../docs/modular-entropy-ablation.md)
defines the controls, conditional conclusions and remaining exceptions.
Every emitted file is independently decoded. The native control must match
the frozen coding replay; partition transplants must match the context maps
emitted by the full reference no-LZ77/cap-32 encoder. These runs do no learning
or transform selection and are not timing benchmarks.

```sh
python3 tools/modular_reference_learning/run_entropy_replay.py \
  --binary build/reference-learning/entropy-partition-bin/reference_learning \
  --coding-study build/reference-learning/coding-replay \
  --output build/reference-learning/new-entropy-study
python3 tools/modular_reference_learning/analyze_entropy_replay.py \
  --study build/reference-learning/new-entropy-study \
  --output build/reference-learning/new-entropy-analysis.json
```

The default runs all 21 controls; `--variants` permits a declared subset and
`--cases` selects `input:policy` pairs. Either `reference_map32` transplant
requires `no_lz77_clusters32` in the same run. Configure the research binary
against the native prepared-stream archives and the pinned reference build.
The combined report additionally takes `--partition-study`,
`--sanitizer-study` and `--validation`; its cohort and byte-identity gates are
specific to the qualified 13-control plus three-control study described in
the report. The [native direct-ANS follow-up](../../docs/modular-direct-ans.md)
adds balanced, high-density, rate and rate/64 data modes, preserving native
tree coding. These variants require `native` in the same run. Use
`analyze_direct_entropy.py` to qualify that follow-up without changing the
original 15-control report.

`native_direct_estimated` and `native_direct_estimated64` use the new estimated
width option and require their matching `native_direct_rate`/`rate64` controls.
Correctness runs record width/cost/config/histogram candidate counts. The
separate `--model-timings` mode disables detailed profiling and records native
model CPU/wall intervals; these individual records are not by themselves a
qualified benchmark. `entropy_model_timing.py run --study RATE_STUDY --output
NEW_DIRECTORY` performs one warmup and five serial interleaved rounds, and its
`analyze` command verifies output identity and timing boundaries.
`analyze_estimated_entropy.py` combines the rate, sanitizer, native storage and
timing qualification. See the [estimated-width report](../../docs/modular-direct-ans-estimated.md)
for the measured scope and remaining integration work.

The original native control and all four direct rate/estimated controls also
compare complete output against the private prepared-stream encoder under its
managed-storage reservation. New records carry `prepared_stream_control`
equality/peak/retained bounds. Direct comparisons are omitted in model-timing
mode. Historical frozen records remain readable. Use
`analyze_prepared_entropy.py` to qualify these integrated paths against the
previous standalone controls and generated Release/sanitizer files; see the
[integration report](../../docs/modular-prepared-entropy.md).

## Public CPU/Metal qualification

`gjxl_modular_metal_workflow_profile` builds with Metal, benchmarks, tests and the
pinned Modular oracle enabled. Its schema 2 records bounded feature/token batches,
the CPU ceiling, actual participation and the joined stream-encoding stage. The
default complete comparison retains public CPU/Metal and the two private forced
feature producers. `--public-only` selects just the public CPU and explicit Metal
routes; `metal-auto` means automatic feature routing within explicit Metal.

```sh
python3 tools/modular_reference_learning/run_metal_workflow_qualification.py \
  --binary build/release/gjxl_modular_metal_workflow_profile \
  --inputs inputs.json --out build/public-metal-check \
  --cpu-participants 14 --public-only --encode-repeats 2 --feature-repeats 3
python3 tools/modular_reference_learning/report_metal_workflow_qualification.py \
  --capture build/public-metal-check \
  --json build/public-metal-check/results.json \
  --markdown build/public-metal-check/report.md
```

The input manifest lists `name`, `path`, `sha256` and `modes` (`encode` and/or
`features`) for each image. Each process checks all output bytes and independently
decodes every encode. Feature trials compare every map with the serial predictor.
Use multiples of two measured rounds for public-only comparisons, four for all
encoder variants and three for feature producers to balance order positions.

The encoder timer includes admission, selection, learning, tokens, entropy,
cleanup and publication. Feature-operation trials partition selected channels by
shape and bounded sample/slice counts, retaining all completed batch owners until
comparison; their memory peak is distinct from the workflow's one-live-local-batch
policy. Selection, planning, independent verification and file I/O are outside
that operation timer. The runner freezes the executable, source, source patch,
inputs and per-call environment; the analyzer also reads historical schema 1
captures without changing their statistics or reports.

## Selected sample-row prototype

`hybrid_profile` accepts an optional `raster|selected|metal|metal-deferred`
argument after the tokenization participant count. It compares the existing
native row path, selected-only CPU rows and Metal rows while reusing the same completed weighted
features. Metal includes native bin lookup and compact-row generation; ordered
deduplication stays on CPU. The workflow prototype uses GPU rows for local
learning scopes and selected CPU rows for shared/global scopes.

`run_sample_rows_qualification.py` rotates all three variants, checks exact
independent decoding and byte identity, and waits for competing codec work.
See [implementation and measured results](../../docs/modular-sample-rows-prototype.md)
for the timing boundary, qualification and current opt-in status.

The follow-up `run_sample_materialization_qualification.py` compares frozen
before/after binaries in alternating order, with warmups, process/power
snapshots, exact decode and byte checks, and overlap exclusion/retry. It measures
the native CPU materialization and ordered deduplication improvement shared by
all row modes. See the [CPU follow-up report](../../docs/modular-sample-materialization.md).

The additional `metal-deferred` mode keeps native ordered deduplication, stores
retained source-row indices, then copies or compacts complete learner columns.
An optional final `rows-on` argument enables replay counters and table-growth
timing; omit it or use `rows-off` for speed comparisons. The paired runner's
`--candidate-metal-mode metal-deferred` compares immediate baseline replay with
deferred candidate replay. Passing the same executable as both binaries isolates
the two replay modes within one build. See the
[deferred materialization report](../../docs/modular-deferred-sample-materialization.md)
and `analyze_sample_preparation_samply.py` for sampled CPU attribution.

`raster-unique`, `selected-unique`, and `metal-unique` enable the optional
geometry/bin uniqueness proof. Proven scopes omit ordered deduplication; every
unproved scope retains native lookup and weight behavior. `metal-unique` uses
deferred replay for its fallback. The profile reports proof time, attempted and
proven scopes, and the number of rows covered. Use the paired runner's
`--baseline-metal-mode metal-deferred --candidate-unique-rows` to compare this
experiment with the committed deferred mode on both CPU and Metal rows. See
[the uniqueness experiment](../../docs/modular-unique-sample-rows.md).
