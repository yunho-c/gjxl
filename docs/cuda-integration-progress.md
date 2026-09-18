# CUDA integration work record

Goal: integrate CUDA with current main, including functional/effort parity,
portable storage and execution-domain accounting, preservation of CUDA's
qualified transport/serializer optimizations, and appropriate verification.
This record tracks incomplete work; it is not an acceptance report.

Base main: `bb7b714e858ffe7668a84c8588745a99b7015f39`.
CUDA baseline: `6e8277b069b79d7ef48b05514f8a160eb6f23648`.
Branch: `integration/cuda-main`, created from main in a separate worktree.
The original CUDA worktree and its frozen artifacts remain available.

## Required work and evidence

- [ ] Reconcile all merge conflicts and inspect automatically merged interfaces.
- [x] Windows/MSVC and Linux standard-library storage bounds, allocation tests,
      language/toolchain rejection tests, and installed C/C++ consumers.
- [ ] Conditional CPU/Metal/CUDA builds, native Rust features and CLI compatibility.
- [x] Main's shared effort, entropy/DC/context-map/ANS policies retained.
- [x] CUDA sparse/compact coefficient ownership, order-population cache, native
      token consumers and packed ANS benefits retained without duplicate work.
- [x] Unified resident input and completed-frame/publication interfaces.
- [x] CUDA fixed-DCT8/uniform low-effort frontend and evaluation-free preparation.
- [x] CUDA prediction-aware DC quantization, decoder-equivalent smoothing and
      reconstruction, including exact mode.
- [x] CUDA nonlinear resident final CfL and dense effort-10 candidate placement.
- [ ] CUDA host/device/pool backing and retained output charged to shared domains;
      finite memory admission, CPU scheduling, trim, failure and batch contracts.
- [ ] Efforts 1–10, four AQ modes, score toggles, distance/maximum-error/size
      controls, odd/tall/large images and concurrent contexts verified.
- [ ] Independent decoder, exact CPU/CUDA comparisons and resident determinism;
      rate/decoded-quality qualification for deliberate policy changes.
- [ ] Whole-public-call warm/cold/batch performance and memory measurements.
- [ ] Portable and real-GPU CI; Metal regression validation; sanitizer coverage.
- [ ] Final audit, reviewable changes and integration-ready branch/PR.

Automatic CUDA selection remains deliberately opt-in, as agreed in the audit.
Do not claim a CUDA/Metal speed ratio from unmatched policies or hardware.
Do not disable resource guards or silently ignore finite-budget options to make
an intermediate build appear complete.

## Current evidence

- Baseline audit: CUDA passed 94 configured tests, 22 pinned-decoder fixtures,
  four workflow conformance cases, and all four CUDA modes on a small independently
  decoded fixture. Those are baseline results, not integrated results.
- Started a no-commit merge of the CUDA baseline into the main-based worktree.
  All 44 initially conflicted files are snapshotted under the ignored
  `build/integration-evidence/conflicted/` directory before edits.
- Integrated Windows CPU libraries, CLI, benchmarks, and test executables build
  with MSVC 19.37 (toolset 14.37.32822), STL 143 update 202305, Release. The
  audited adapter covers vector/string growth and MSVC hash-table sentinels and
  bucket storage. The GNU adapter now builds under Ubuntu 24.04 GCC 13.3.0,
  headers `20240904`, with runtime libstdc++6 `14.2.0-4ubuntu2~24.04.1`.
- The CPU configuration has passing evidence for all 108 applicable tests
  across the full run and an isolated installed-consumer rerun. The latter
  passed in 169 seconds; its MSVC timeout now accommodates standalone header
  compilation. Later CUDA/accounting edits still require regression checks.
  AC-search host validation passes its reservation/failure/reuse cases after
  restoring separate Metal buffers and limiting packed arenas to CUDA.
- Main's effort/DC/entropy policy and writer structure are combined with
  CUDA-native dense/compact/sparse ownership and token consumers. Native AC
  arrays and the immutable coefficient-order population payload now use managed
  backing. Score publication uses main's transactional ownership contract.
- The full CUDA configuration builds with CUDA 11.8 / SM86. Its first CUDA-only
  suite passed 32/34 tests. The resident host-allocation regression was fixed;
  both outstanding failures now reach the unsupported CUDA workflow admission
  route. New completed-frame tests pass across producer reuse/destruction.
- Nonlinear final CfL passes 1,720 exact scalar CPU map comparisons, alongside
  576 fast-path comparisons and invalid-input/reuse cases. CUDA memcheck passes
  the smaller 24 fast + 360 nonlinear comparison set with zero errors.
- Resident input supports supplied and generated sources, with padded/strided,
  odd/tall, numeric validation and failure-atomic publication tests passing.
- Uniform initial fields, omitted AC-search masks and evaluation-free CUDA
  preparation pass 24 low-effort cases. Coefficients match a scored zero-update
  reference; reduced preparations allocate less device storage. The full sparse
  resident fixture still passes after these changes.
- CUDA buffers now carry allocation-domain tickets. Completed reusable buffers
  live in a bounded explicit cache, retaining their tickets while idle. Driver
  free-page retention is disabled; driver granularity/internal overhead remains
  outside the managed-backing contract. Domain-specific trim and failed
  admission can reclaim idle backing. Tests pass for finite reservations,
  injected physical allocation failure, domain isolation, exact-size reuse,
  trim boundaries and lifetime after backend destruction. CUDA's existing
  asynchronous-use, shared-pool and trim tests also pass.
- CUDA host metadata/staging uses managed backing with preserved admission
  errors. Shared allocation recipes now cover resident input, Butteraugli,
  resident AQ, exact AQ and frame-only AQ. Resident plans pass 32 shape/policy
  cases and 35 injected host-allocation checkpoints; compatibility plans pass
  16 cases, including uniform/adaptive frame-only initialization and DC policy.
  CUDA AC-search plans include packed-arena alignment.
- Prediction-aware DC quantization, gradient/weighted predictors, extra DC
  precision and smoothing now have CUDA kernels. All 378 direct CPU/CUDA
  quantization/reconstruction/smoothing comparisons pass exactly. Resident
  low-effort tests pass 96 cases, including scored/evaluation-free byte equality
  across the DC policies. Direct exact-frame reconstruction now applies DC
  smoothing and passes comparisons against the CPU for four DC policies.
- Public admission selects CUDA-specific storage plans for all four modes.
  Compatibility routes conservatively compose the shared CPU owner bound with
  concrete CUDA owners; no guard is bypassed. Maximum-throughput retains CUDA's
  resident input and reusable frame-only preparation across target-size retries.
  The integrated CUDA AQ workflow and benchmark CLI tests now pass.
- The shared public admission suite passes with CUDA, including finite limits,
  C/C++ shared domains, underplan and allocation failure recovery, target-size
  retries, and 36 batches across image sizes, routes, worker counts, cache
  retention and forced retirement. Later generated-input changes still require
  rerunning this evidence.
- The Windows CUDA configuration passed 147/148 tests in its full run. The
  remaining C API resident-input handoff guard was corrected; the focused
  CUDA admission rerun passes, including the generated-input path. CPU admission
  is now registered on every platform and passes separately on Windows/Linux.
- Linux passed 107/108 tests initially; its distribution `jxlinfo` 0.7 prints
  different metadata labels. Supporting those exact spellings preserves all
  value checks and makes the smoke rerun pass. Together with CPU admission,
  there is passing evidence for all 109 applicable Linux tests, including
  storage/compiler guards and installed consumers. This is aggregated evidence,
  not a claim that the final source snapshot has had another full run.
- All 23 integrated pinned-decoder fixtures and four public workflow cases
  pass. The stale impulse hash was refreshed only after an independently built
  main `bb7b714` codec/serializer reproduced it. All 23 integrated fixture
  streams match that main control byte-for-byte. The control uses main's source
  plus the GNU allocation adapter and portable test launcher, not CUDA writers.
- A 150-case public decoder matrix covers efforts 1–10, four CUDA modes and
  three image shapes. All decoded pixels are finite; all 30 exact-mode CUDA
  outputs match their integrated CPU controls byte-for-byte. This is synthetic
  smoke qualification, not photographic rate/quality acceptance.
- Sixty-four mixed-transform resident reconstructions match the stored-frame
  CPU consumer with DC policies, smoothing and loop filters toggled. Existing
  sparse/dense transition, native-population and completed-lease tests pass.
- DC memcheck completed all 378 CPU/CUDA comparison cases with zero errors.
  Racecheck completed an 18-case subset spanning both 256-block boundaries,
  prediction modes and precision bits, with zero errors/warnings. Completion
  files verify execution despite sanitizer console-handle behavior. The first
  full racecheck run was stopped for its instrumentation cost and is not counted.
- Explicit deferred transform metadata now requires successful Reconfigure
  before final CfL, policy preparation or evaluation. Initial AC-search data
  remains available. Four direct deferred/eager, mixed/DCT8, scored/encoding-only
  cases pass, alongside the public CUDA AQ and compatibility storage tests.
- Windows Rust tests pass with CPU (11) and CUDA (12, including a real device).
  Both feature configurations pass clippy with warnings denied. The smoothing
  enum conversion now handles MSVC's signed bindgen constants. CI's last text
  conflict is resolved; native/Rust matrices select the audited toolchains,
  and a dispatchable Windows SM86/CUDA 11.8 qualification job is prepared.
  YAML parses and the existing-install MSVC selection script runs locally;
  hosted CI and the installer fallback have not yet run.
- Initial whole-call timing screen, RTX 3060 Laptop, distance 1.9, automatic
  CPU workers, two warmups/five samples: effort 1 improves from 76.7 to 41.8 ms
  at 1919x1079 and 178.6 to 123.1 ms at 3839x2159. Default effort 7 changes
  from 64.4 to 74.5 ms and 247.2 to 276.0 ms; effort 8 from 74.1 to 143.9 ms
  and 330.2 to 550.1 ms. These are different-policy comparisons.
  With ordinary DC rounding, gradient prediction and smoothing disabled,
  integrated effort 7 measures 64.6/248.0 ms (seven samples), close to the
  old CUDA baseline. This points to new DC-policy work rather than broad
  accounting overhead; matched rate/quality and repeated paired runs are still
  required. The benchmark now accepts and reports the public DC overrides.
- Full C++23 builds and suites pass on Windows (108/108) and Linux (109/109),
  including installed consumers. Linux Rust 1.98.1 passes all 11 wrapper tests,
  formatting and clippy with warnings denied; its native build uses the same
  audited GCC 13.3 headers/runtime as the C++ suite.
- Dense DCT32-family anchors now have independently enumerated fused-kernel
  coverage: 338 cases, 41,008 candidate descriptors, 338 separate forward/loss
  references and 1,014 repeated fused batches pass. Full CPU/GPU grid search
  and retained sparse/dense policy transitions also pass.
- Public CUDA DC defaults/overrides pass in all four AQ modes with gradient
  and weighted prediction. Final-score collection preserves bytes, exact-mode
  outputs match CPU controls, and batches retain each image's resolved policy.
- A 160-output photographic sweep uses four original, hash-verified CUDA
  qualification images at distances 0.5/1.2/3/6 and efforts 1/4/7/8/10. All
  independently decoded images are finite. Pinned Butteraugli measurements
  show material low-effort quality tradeoffs and some higher-effort outliers;
  successful decoding is not treated as rate/quality acceptance. Sparse
  four-point interpolation is exploratory and requires direct matched-size
  checks before drawing acceptance conclusions.
- The qualification driver now builds for CUDA on Windows, reporting complete
  public-call times, process working set, managed-domain counters and explicitly
  device-wide CUDA memory snapshots. Forty alternating baseline/integration
  processes plus eight integrated pressure processes pass (seven calls each).
  With legacy DC controls, median paired warm-time ratios are 0.987 small,
  0.936 odd 1080p, 0.973 odd 4K, 0.972 native photo, and 0.957 mixed batches with
  two simultaneous callers. Cold samples and every pair are retained separately.
  These compare one GPU and one workload family, not Metal/CUDA performance.
- In the 4K pressure cases, peak managed backing is 2.146--2.154 GB against a
  4.983 GB conservative reservation, Windows peak working set is 428--429 MB,
  and post-call device usage rises by 1.980 GB. Device usage returns to the
  initial snapshot after trim; post-trim managed domains are empty in every
  case. Device snapshots include other device users and are not process peaks.
  One/three CPU-participant caps and tight/full batch budgets pass. The existing
  qualification controller's ten tests also pass after the driver extensions.
- GitHub's existing main Rust job on `macos-26` succeeds at the pinned main
  revision (run 35299888139). The connector can read its status even though
  the local `gh` CLI is unauthenticated. Integrated native/Metal CI still needs
  to run; historical main CI is not evidence for the merged source.
- Interface review caught a C API merge drift in quality-to-distance arithmetic.
  Main's double intermediates are restored with an explicit float return cast,
  preserving its rounding while satisfying MSVC warnings. The canonical quality
  80 check now requires the exact public float 1.9. Both focused C and C++ API
  tests pass; Rust checks still need to run after this correction.
- Thirty-two additional photographic CPU/exact-CUDA controls reveal one byte
  mismatch among sixteen pairs: keong, effort 8, distance 1.2 (35,375 CPU bytes,
  35,365 CUDA bytes; decoded Butteraugli 1.43334/1.43010). A per-update probe
  reproduces identical initial fields and raw quantization through two updates.
  Before the third update, block-map differences peak at 0.000262; four raw
  quantization values then differ by one. This is a threshold-crossing effect
  exposed by main's additional refinement, not an accepted universal byte-match
  claim. Fixed-field coefficient contracts and this end-to-end numerical case
  must be distinguished during final qualification.
- The refreshed complete Windows CUDA suite passes all 150 tests, including
  public admission, DC policies, benchmark CLIs and the installed consumer
  (918 seconds). Hosted Rust checks pass on Linux and macOS at `9cf3e65`.
  Native macOS C++20 reached testing after the Metal component setup fix.
  C++23 exposed a test's missing direct `<exception>` include; the include is
  now explicit and its focused Windows C++23 check passes.
- Hosted Linux native C++20/C++23 checks pass. Metal runtime testing revealed
  an integration regression: an omitted host pixel mask was passed with nonempty
  geometry. Main's empty-view contract is restored. The shared AC-search test
  now distinguishes CUDA arena capacity from Metal's separate buffers; worker
  fault tests retain explicit four-participant coverage while limiting automatic
  fault positions to the runner's available worker count. Scalar Metal pipeline
  failures are being compared with pinned main on the same runner, not suppressed.
- External distance bracketing produced 40 additional independently decoded
  photographic comparisons; 36 meet a 0.2% byte-size gate. Among gated cases,
  median decoded Butteraugli ratios (integrated/baseline) are 1.183 at effort 1,
  1.274 at effort 4, 1.029 at effort 7, 0.996 at effort 8 and 1.005 at effort 10.
  Efforts 1/4 have six gated pairs each; the others have eight. The effort-4
  maximum is 1.548 and effort-7 maximum 1.129. Four unmatched cases remain
  explicitly excluded. These compare changed default policies; low-effort
  tradeoffs and the exact-mode threshold finding still require adjudication.
- CUDA stage profiling now brackets image-primitive and AC-candidate submissions
  with device events. Primitive tests pass for output preservation, repeated
  snapshots, finite diagnostic reservations, lifetime and injected failures.
  AQ/dispatch instrumentation and whole-workflow profile planning remain open;
  the public profiling gate is deliberately still closed. See
  [CUDA profiling integration](cuda-profiling.md) for timestamp semantics.
  Profiled AC search matches its ordinary result, including an omitted host mask.
  Memcheck reports zero errors with an explicit child completion marker. All
  nine focused pipeline/search/CPU-launch regressions pass after the Metal fixes.
  Hosted Rust checks pass on Windows, Linux and macOS at `53dfbeb`.
- The hosted Mac is an Apple Paravirtual device, not a physical Apple GPU.
  Pinned main reproduces the scalar DCT and worker-fault failures on that same
  runner (job 105523511997). The scalar DCT32 shader now distributes its existing
  per-element arithmetic over the pipeline's supported thread count, preserving
  each dot product's order; this change is awaiting hosted shader/runtime tests.
  Worker fault positions now respect the hardware cap for explicit limits too.
  The quality benchmark retains main's `metal_aq_mode` JSON field alongside the
  generic alias, and the effort-9 CLI assertion follows main's rate-optimized
  policy. Timestamp-graph tests explicitly skip when the device rejects stage
  counters; unsupported-profile output preservation remains tested. Such skips
  do not qualify Metal timestamp behavior on physical hardware.

- Resident CUDA AQ stage capture now covers reference preparation, initial
  quantization, resident policy setup/evaluation and optional sparse packing.
  The encoding-only path preserves device-owned fields. Eighty fresh/reused
  byte-and-score comparisons pass across two fixtures, efforts 1/4/7/8/10,
  both resident modes and score toggles. Concurrent/nested capture tests pass
  with finite diagnostic reservations, backend/thread isolation and retained
  ownership checks; the focused profiling/AQ suite passes all three tests.
  Public GPU-profile admission remains closed pending full workflow planning
  and dispatch instrumentation remains unfinished.
- At `1d1530b`, hosted Linux C++20/C++23 and all three Rust jobs pass. Both
  hosted Mac modes now pass scalar DCT and AC search; remaining failures expose
  missing Metal color-tile validation, timestamp capability assumptions,
  benchmark timing assumptions and numerical comparisons. Metal preparation
  and reconfiguration now reject crossing strategies before GPU work; hosted
  validation of this fix is pending. Remaining numerical checks are being
  compared with pinned main, without relaxing tolerances.
- The Windows installed-consumer failure came from switching the working
  Ninja/MSVC toolchain to a different Visual Studio generator. The fixture now
  reuses the parent generator, compiler and build/resource tools. Its complete
  local installed-header/consumer check passes in 68 seconds; hosted rerun is
  pending. Serializer failure diagnostics now distinguish byte, participant,
  accounting and fallback-domain failures.

- The diagnostic CUDA public-workflow entry point now supports stage profiles
  with normal finite-domain admission. Its storage recipe derives reference,
  initial, optional AC, policy setup/evaluation and sparse-pack counts from
  the call graph, and includes nested capture, retained recordings and snapshot
  overlap. Eighty fresh/reused workflow comparisons preserve bytes and full
  summaries under the computed bound; all 55 diagnostic allocation boundaries
  in an effort-7 workflow preserve outputs on failure. Under-admission rejects
  before GPU work. Four hundred geometry/policy planning cases allocate no
  managed backing. Dispatch records and benchmark profile export remain open.
  The earlier capture test also passes memcheck with zero errors and an
  explicit child completion marker. All nine focused CPU/CUDA admission,
  storage, DC-policy and profiling checks pass (110 seconds), followed by
  the strengthened workflow check for default-domain allocation escape.

- Hosted CI at `3a8da47` passes Linux C++20/C++23 and all three Rust jobs.
  Metal's five remaining failures reproduce at that revision. Mixed-frame
  fixtures now keep every transform within its color tile; resident AQ tests
  check atomic rejection when timestamps are unavailable and keep ordinary
  execution/materialization coverage. The three changed test sources pass
  MSVC syntax checks. The next hosted diagnostic runs pinned main with only
  the scalar-DCT launch-width portability patch, allowing numerical controls
  to reach execution on the paravirtual GPU. Numerical tolerances are unchanged.

- CUDA dispatch profiling now records all 134 kernel launch sites, launch
  geometry and invocation IDs; dispatch mode adds per-launch event intervals.
  A source audit verifies unchanged kernel bodies, arguments and configuration.
  Stage/dispatch primitive and resident-AQ checks pass, including 160 exact
  byte/score pipeline pairs and 160 finite-domain workflow comparisons. All 678
  diagnostic allocation boundaries in the effort-7 workflow preserve outputs.
  Additional primitive sweeps fail between dependent launches and during
  snapshot publication, then verify same-backend recovery. Memcheck reports
  zero errors with a completion marker after those checks. The nine focused
  admission, storage, DC-policy and profiling tests pass in 126 seconds.
- Ordinary whole-call profiling-overhead controls use a separate build of
  `6c90d83` with the same compiler and driver source. Sixty-four alternating
  processes cover small images, odd 1080p/4K, a photo and concurrent batches;
  every output signature matches. Median paired ratios range from 0.945 to
  1.026. The initial 1080p effort-1 +2.6% result does not reproduce in a longer
  24-process, 31-sample follow-up (median ratio 0.9875). These are noisy
  single-device overhead checks, not claims of general speed improvement.
- At `6c90d83`, hosted Metal C++23 passes the corrected resident AQ,
  completed-frame and AQ-storage tests. Its remaining failures are two
  numerical comparisons plus a cache-admission timeout. Main with the scalar
  DCT portability patch reproduces the identical maximum-throughput pixel-mask
  error (0.000419617). Its reconstruction test uses the old mixed fixture and
  instead reaches a later initial-quantization pixel error (0.000164032), so it
  does not yet adjudicate the integration's DC mismatch. The next control uses
  the same valid reconstruction fixture. New diagnostics retain exact DC and
  numerical checks; cache subcases report progress and retain their internal
  5-second admission waits while allowing 180 seconds for repeated backend
  creation (the prior successful hosted run took 43.4 seconds overall).

- CUDA benchmark profile export now supports `--gpu-profile stage|dispatch`
  and `--gpu-profile-output` through the shared Metal/CUDA schema-4 writer.
  Diagnostic warmups/samples preserve ordinary CUDA bytes and full summaries;
  ordinary timing runs retain their separate path. CLI tests cover both AQ
  modes, efforts 1/7/8/10, final scores, odd PFM input, all bundled workloads,
  DC overrides, repeated output replacement, invalid options and failed final
  publication. The three focused benchmark/writer tests pass on CUDA; shared
  writer tests also pass with Windows C++23 and Linux C++20. JSON numbers are
  locale independent; failed writes preserve outputs and clean staging files.
  The extracted Metal benchmark passes an MSVC syntax check after correcting
  two existing designated-initializer ordering errors. The full Windows CUDA
  build at `7154305` passes all 155 tests in 518 seconds, including the installed
  consumer. CUDA's semantic stages still identify whole submissions, while
  dispatch mode provides individual launch timings. The explicit GPU
  qualification workflow now retains the previously validated profiling
  lifetime/failure-recovery memcheck and its required completion marker,
  alongside the DC memcheck/racecheck gates.
- At `27c4344`, both hosted Metal modes pass 147/149 tests, including cache
  admission (64 seconds on C++20, within the revised outer deadline). The main
  control with the scalar-DCT portability patch and identical valid mixed
  reconstruction fixture reproduces both numerical failures exactly: channel-0
  DC block (4,5), 3979 versus 3978, and maximum-throughput pixel-mask error
  0.000419617. The latter retains identical frames and codestreams. These are
  inherited numerical checks on the paravirtual GPU, not evidence of a CUDA
  integration regression; they remain unresolved, with tolerances unchanged.
  Hosted Windows and Linux C++20/C++23 and all three Rust jobs pass at this
  revision.

## Outstanding integration risks

- Further isolated CUDA controls identify a Windows host-library difference:
  MSVC's `std::cbrt(float(0.0037930732552754493))` returns bits `3e1fb276`,
  while CUDA's fixed inverse-color constant is `3e1fb275`. Matching the host
  value, CPU-order Gaussian filters, double inverse-DCT accumulation, paired
  Gaborish sums and serial block reduction produces identical reconstructed RGB,
  raw quantization and codestream bytes in 20 comparisons: four photographs,
  zero through four AQ updates, prediction-aware DC with smoothing. Remaining
  metric block error is at most 3.6e-7. Changing only the color constant does
  not fix the original decision discrepancy. These are diagnostic linked-object
  overrides, not production changes or a performance qualification. The next
  implementation must isolate CPU-compatible arithmetic to exact mode, obtain
  platform-dependent constants from the host, preserve ordinary resident
  arithmetic and account for any added storage. No CPU fallback is introduced.

- The later Metal resident-CfL failure exposes a separate functional defect:
  preparation validates but discards its supplied field/DC value, deriving the
  supposedly invariant map from the next evaluation instead. The candidate fix
  snapshots the field in existing policy scratch, derives its quantizer before
  CfL, then uses the evaluation quantizer for coefficient coding. It adds no
  device allocation or submission. Profile planning reserves the extra selector
  dispatches. Regression coverage poisons the caller's field after preparation
  and also evaluates host-provided raw quantization. Hosted validation is pending;
  Windows validates the changed test syntax but cannot compile Metal internals.
- A separate CUDA diagnostic reproduces CPU Gaussian accumulation order on the
  device. On the keong fixture with CPU-reconstructed pixels, metric block error
  falls from roughly 0.0002 to at most 3.6e-7. With CUDA reconstruction, residual
  error still crosses AQ decision boundaries, changing 18 raw values after two
  updates. The experiment is retained only in ignored diagnostic artifacts;
  production convolution kernels and compiler flags remain unchanged.

- At `156ee67`, hosted Metal C++23 passes 149/150 tests. Exact DC basis
  constants remove the earlier DC mismatch, and a precise mask logarithm
  clears the maximum-throughput pixel-mask check without changing tolerances.
  Reconstruction now reaches a later resident quantizer/invariant-CfL check;
  additional diagnostics distinguish its numerical and resource assertions.
  Both Linux modes and all three Rust jobs pass at this revision; remaining
  native jobs are still running. Physical Apple GPU validation remains open.
- The keong photograph's exact-mode discrepancy is isolated further. With
  legacy DC settings, the original CUDA branch (`6e8277b`) and integration
  produce identical diagnostic results through four AQ updates, including
  CPU-identical raw quantization. Under the new DC policy, supplying CPU
  reconstructed pixels to the CUDA metric still leaves block-feedback errors
  around 0.00018–0.00023 before quantization decisions diverge. A separate
  CUDA build with implicit FMA contraction disabled still changes four raw
  values after three updates and seven after four. That compiler flag is not
  adopted. CPU and CUDA convolution accumulation order differs and is the
  next numerical investigation; no exact-mode contract is relaxed.

- Broaden public policy/DC/low-effort coverage to all efforts, controls and
  representative images. Verify dense effort-10 behavior against independent
  references and qualify the new effort-8 quality/performance tradeoff.
- Qualify storage recipes on larger images and all optional policies; tighten
  conservative compatibility bounds only with allocation evidence. Check the
  explicit CUDA cache's complete-call performance and physical memory behavior.
- Verify Metal behavior on appropriate hardware.
- Finish hosted CI execution, GPU diagnostics parity, broader qualification
  and matched performance/memory measurements. Merge checkpoint `7449d6c` is
  published as draft PR #29; it is not ready to merge. The first native macOS
  CI run reached shader compilation but lacked Xcode's separate Metal compiler
  component. CI setup now installs that component before testing the build.
