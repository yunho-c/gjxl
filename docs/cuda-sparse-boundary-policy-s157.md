# S157: sparse boundary contracts and selection-policy measurements

## Outcome

S157 qualifies the S156 native sparse frame boundary against injected failures
and concurrent prepared contexts, then measures the small/dense and retained
regimes missing from S156. **It does not promote the overlay or select a default
sparse policy.** Production CUDA source remains at S152 (`2118ccb`); the starting
repository commit is the S156 evidence milestone `59f35fe`.

The useful results are:

- 89 CTests passed, including the new failure and concurrency fixtures.
- Each complete failure-fixture run covers 217 injected failures, 217 rejected
  retries on invalidated contexts, and 119 successful controls.
- Each complete concurrency-fixture run covers 32 prepared contexts and 192
  evaluations, compared against 24 dense controls.
- Fully instrumented host AddressSanitizer builds passed both fixtures.
- Portable Compute Sanitizer 2025.2.1 passed eight integrated checks and two
  isolated race checks without errors, hazards, or reported memory leaks.
- All 3,528 synthetic timing/oracle calls and 48 fresh frozen-corpus calls were
  byte-exact against their controls. Synthetic controls are in-process dense
  results, not independent frozen oracles.
- Low-detail retained 4K evaluation plus serialization improved 13.63–18.79%
  with int32 storage and 8.526–8.549% with compact storage in the two repetitions.
- Dense inputs do not justify forced sparse storage. The existing late fallback
  itself regressed 1.51–4.77% on the measured int32 dense cases. Selection should
  happen before the extra packing submission and header readback.

The legacy CUDA11.8 racechecker produced a warning/hangs that the newer tool did
not reproduce on unchanged executables. These rejected and partial results are
preserved, not counted as clean qualification. The exact cause is not established.

## Implementation scope

The experiment lives in `build-cuda-ninja/profiles/s157-artifacts/source`, copied
from the frozen S156 overlay. Main-tree `src`, `tests`, and `CMakeLists.txt` were
not changed. There is no new compatibility layer or dense-expansion adapter.

The native owner, six-way dense/sparse typed dispatch, coefficient consumers,
and 256-thread GPU packing algorithm are unchanged from S156. Added diagnostic
controls exercise the sparse boundary inside the real resident evaluation path.
They are not a proposed production fault-injection API and require a quiescent
prepared object/backend before arming.

The only ordinary-path behavioral change is routing the one-element sparse
payload copy through `CopyDeviceToHostBatch`, allowing its real enqueue/drain
path to be tested. Disabled diagnostic atomic checks remain in allocation and
batch-copy paths. Remove those controls before any source promotion.

All seven inspected S157 timing/ordinary/host-ASan executables share all eleven
GPU modules byte-for-byte. After normalizing only two source-derived anonymous
symbol IDs, the sparse kernels are instruction-identical to S156, with int8/
int16/int32 register counts 30/32/28, 68 shared bytes, and no stack/local storage.
All resource records match S156. The pre-existing `PrepareQuantNormsKernel`
instruction sequence differs from the S156 ordinary build; every other module
matches after symbol normalization. The precise rebuild cause is unproven.
All timing comparisons use alternatives in the same executable and therefore
the same generated GPU code.

## Boundary qualification

The failure fixture prepares a 65x67 DCT8 frame. Scored and unscored resident
policies exercise sparse int8, int16, and int32; ordinary `Evaluate` exercises
int32. Each configuration tests cold and already-used contexts. Device allocation
failure applies only to the cold context because the sparse header buffer is
retained afterward.

The sixteen fault points cover device/header/value allocations, submission,
completion, callback failure after work is queued, header readback before/after
enqueue and after completion, invalid count/offset/frame metadata, and malformed
zero payload. Allocation injections simulate `bad_alloc` or CUDA allocation
failure at selected sites; they are not exhaustive allocator interception or
physical out-of-memory tests.

For every failure, quant fields, distance maps, scores, quantizer parameters,
RGB, the pre-existing frame/codestream, coefficients, and dense-owner pointer
remain unchanged. A subsequent call on the invalidated context fails without
new backend allocations or submissions. Fresh independent contexts still succeed.
Enqueue-error injections use actual queued copies; the existing batch-copy path
drains those copies before returning the synthetic error.

Concurrency uses four geometries: 65x67, 257x17, 513x519, and 129x131. It crosses
shared/independent backends, pooled/nonpooled allocation, and forced wide/compact
sparse modes. Four host threads each own a prepared object; six rounds alternate
scored policy, unscored policy, and ordinary evaluation. Outputs, coefficient
values, RGB, and codestreams match dense controls. Changing the thread-local
diagnostic mode after preparation does not change a prepared object's policy.
Independent backends have independent streams; these are not claimed to be
independent CUDA driver contexts.

The new sanitizer matrix checks all kernels for memory/initialization errors;
race/synchronization checks select all new `SparseAc` launches. Existing kernels
are not claimed to have received new full race instrumentation. No suppression,
blocking-launch option, or launch-count limit is used in that matrix.

The isolated pack fixture additionally enqueues four nonblocking-stream launches
before waiting for any of them. It checks 3 coefficient counts (65, 65537,
811200), 7 sparsity patterns, 3 widths, and 4 workers: 252 cases per run. All
input, payload, header and counter guards, absolute offsets, payload coverage,
and logical coefficients pass normally and under the newer racechecker.

## Legacy sanitizer discrepancy

CUDA 11.8's bundled tool identifies as Compute Sanitizer 2022.3.0. Its full
concurrency racecheck stopped progressing after the first of eight configurations
and was deliberately interrupted after 5m19s. An isolated int8 retry reported
60 byte hazards between write PC `0x190` and read PC `0x710`, then stopped making
worker-step progress. That run was interrupted after 2m36s and retains its
exit-86 journal and one-warning summary.

The matching machine instructions are a shared-prefix initialization and final
prefix read. Two unconditional `BAR.SYNC` instructions separate them. This rules
out an obvious missing source barrier; it does not by itself establish the exact
cause of the instrumented run's behavior. NVIDIA documents racecheck as shared
memory hazard detection and warns against dismissing warp-level warnings.
[Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

The portable newer tool was fetched from NVIDIA's CUDA12.9.1 redistributable
manifest, with the published archive size/SHA256 checked before extraction.
It reports version 2025.2.1.0, build 35969825. The unchanged isolated executable
completed all 24 evaluations with zero hazards in about ten seconds; the full
unchanged fixtures then passed the eight-check matrix. No compiler, driver,
clock, power, affinity, priority, firewall, or security setting was changed.
[NVIDIA redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)

The old tool also stopped progressing near the end of the isolated pack test
(80 of 84 worker progress groups printed); it was interrupted after 2m27s, with
no hazard printed. The newer tool completed all 252 cases. A legacy
launch-count-one diagnostic completed, but instrumented only one sparse launch
and is not full race qualification. These comparisons establish tool-version
dependence, not a vendor-confirmed diagnosis or proof of a particular fixed bug.
NVIDIA's release history includes several barrier/racecheck fixes, but none is
asserted to explain this case.
[Compute Sanitizer release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)

Only exact, identity-verified owned test children were stopped. Their logs and
journals remain rejected. No admin/firewall blocker was observed. A discovered
v12.2 directory lacked the sanitizer executable; the unsuccessful launch created
no child and left one explicitly excluded empty log, not a successful test.

## Timing protocol

One executable compares two dense labels against two candidate labels. Each
four-call round rotates the ABBA positions. There are four warmup and eight
measured rounds per process, with two shuffled/reversed repetitions. The primary
statistic is the median of eight within-round candidate-mean minus dense-mean
differences; percentage differences are computed per round before taking their
median. They are not ratios of independently pooled medians.

There are 56 whole-workflow processes and 16 retained-context processes: 2,304
measured calls, 1,152 warmup calls, and 72 extra dense oracle calls. All return
identical codestreams; whole-workflow summaries and retained quant fields/scores
also match. Every timing job is sequential and isolated from other GPU jobs,
compilation and binary inspection. The 6,912 surrounding read-only NVML enforced
power-limit observations are all 40,000mW. This does **not** establish constant
actual power, GPU clocks, temperature, or system load.

Whole-workflow cases use widths 65/257/513/1025 and heights width+6, a retained
backend but fresh pipeline per call, effort 7, eight CPU workers, and no final
score. Three fixed-seed input/quality patterns are tested:

- Low detail: RGB 0.2 plus noise amplitude 0.01, distance 1.2; only
  0.0013–0.0064% of logical AC slots are nonzero.
- Ordinary noise: amplitude 1, distance 1.2; 39.7–45.6% nonzero.
- Near-dense: amplitude 1, distance 0.01; 97.3–98.2% nonzero.

Every case compares forced sparse with dense at each width policy. The near-dense
513/1025 cases additionally compare S156's automatic late fallback with dense.
Input generation/oracle construction, checking, and result destruction are
outside measured intervals; pipeline construction and normal work are inside.

Retained cases use two live prepared contexts, a fixed DCT8 grid, low-detail
amplitude 0.01 input, initial quant field 0.8, and one unscored resident policy
update. Fresh caller frame owners are used for each call. Preparation and caller
output allocation are excluded; evaluation and default-options serialization
are measured separately and together. This is not the full adaptive-strategy
workflow, does not measure dense retained inputs, and does not establish typical
natural-image speedups. The two simultaneously retained contexts also change
the memory footprint relative to a one-context application.

## Results and selection implications

All table entries are elapsed-time percentage changes from the same-process
dense control, spanning the two repetitions; negative is faster.

Low-detail whole workflow:

| Input | int32 | Compact |
| --- | ---: | ---: |
| 65x71 | -1.23 to +1.71% | -4.77 to -4.16% |
| 257x263 | -13.87 to -12.12% | -7.86 to -7.29% |
| 513x519 | -17.97 to -16.02% | -15.43 to -11.77% |
| 1025x1031 | -16.82 to -14.53% | -9.86 to -8.23% |

All twelve processes for low-detail inputs at least 257 wide favor sparse in all
four candidate-label/control-label median comparisons. The smallest wide case
does not support a reliable improvement. The current automatic size floor near
512x512 is therefore conservative for this low-detail input, but these four
sizes do not establish a universal crossover threshold.

Low-detail retained evaluation plus serialization:

| Input | int32 | Compact |
| --- | ---: | ---: |
| 65x71 | -2.15 to -1.29% | -2.90 to -0.53% |
| 513x519 | -24.89 to -23.83% | -15.46 to -14.16% |
| 1919x1079 | -23.30 to -22.07% | -12.26 to -11.21% |
| 3839x2159 | -18.79 to -13.63% | -8.549 to -8.526% |

At retained 4K the paired median savings are 16.56–23.34ms wide and
9.76–11.33ms compact. Evaluation alone improves 30.24–32.26% wide and
9.69–12.43% compact. Serializer results are not uniformly better: one wide
repetition's serializer regresses 8.46%. One compact 4K repetition also has one
cross-label comparison at +4.26%, despite the primary paired statistic favoring
sparse. Treat the narrow ranges of primary percentages as observed repetitions,
not confidence intervals or evidence of invariant performance.

The retained low-detail 4K native host owner is 4,665,608 bytes wide versus
106,168,320 dense; compact is 4,665,602 versus 26,542,080 bytes. These are native
coefficient-owner allocations, including masks/offsets and fixed dense tails,
not wire bytes or whole-process/device memory. Sparse adds a retained 4,665,604-byte
GPU header/count buffer at this extent; the payload reuses existing workspaces.

The ordinary-noise cases are mixed: sparse changes range from -4.28% to +9.51%,
with several direction reversals between repetitions. A wire-size reduction
alone is not a sufficient performance predictor for native sparse consumption.
At near-dense 1025, forced compact sparse is consistently slower (+2.54–5.84%);
the wide result ranges from -0.11% to +3.42%.

All automatic high-density cases select dense fallback, preserving the original
typed GPU source. Nevertheless the pack kernel, header owner/allocation and
first readback have already happened:

| Near-dense input, automatic fallback | int32 change | Compact change |
| --- | ---: | ---: |
| 513x519 | +1.51 to +3.78% | -3.69 to +2.21% |
| 1025x1031 | +1.73 to +4.77% | -1.73 to +2.24% |

The wide fallback penalty is 0.55–1.63ms at 513 and 1.94–5.24ms at 1025. Compact
results are noisy rather than reliably beneficial. Do not promote S156's
"sparse bytes below 80% of active dense bytes" late gate as a finished default.

The next experiment should produce an integer nonzero-count/density statistic
in an existing final coefficient packing/compact pass and return it with an
already-required small readback. Selection can then precede the sparse header
allocation, extra submission and first readback. Compare that fusion against a
separate-statistics-pass control; measure the cost on dense inputs and keep a
conservative small-input gate. This is a proposed experiment, not implemented
or proven faster in S157. Keep native sparse consumers and remove diagnostic
selectors/fault hooks only after a qualified default policy is measured.

## Evidence and reproducibility

The artifact root contains source/build-input snapshots, ordinary and fully
instrumented host builds, seven executable native-module inspections, timing
protocols/raw samples, control files, failed sources/executables, tool archive
and hashes, and all 123 job journals (117 accepted, six rejected). The one empty
missing-tool log is separately excluded. `diagnostic_notes.md` records exact
owned-process interruptions, fixture corrections and qualification overlap.

The fresh corpus check reuses the frozen S156 harness linked against S157
libraries: flower500, paddedHD, flower2000, padded4K, flower3200, and Keong4K;
wide/compact, duplicate dense/forced-sparse calls, 48 complete encodes. All input
and reference hashes match the frozen S153 case manifest. No timing conclusion
is drawn from these correctness-only calls. See the
[S156 report](cuda-native-sparse-frame-s156.md) for its separate natural-image
paired timings and broader frame/serializer qualification.

Key scripts under `build-cuda-ninja/profiles`:

- `s157_prepare.py`, `s157_build.py`, `s157_contracts_v2.py`, `s157_asan.py`.
- `s157_sanitize.py`, `s157_sanitize_v2.py`, `s157_sanitize_v3.py` and
  `s157_pack_diagnostic.py`; the first two and the legacy pack comparison
  intentionally retain unsuccessful historical runs and cannot be blindly
  rerun into existing exclusive journals.
- `s157_fetch_sanitizer.py`, `s157_native.py`, `s157_oracles.py`.
- `s157_timing.py`, `analyze_s157.py`, `verify_s157.py`, `freeze_s157.py`.

From the repository root, `python build-cuda-ninja/profiles/verify_s157.py --frozen`
checks the complete frozen inventory, S156 predecessor hashes, build inputs,
tool/binary pins, rejected-job set, oracles, analysis reproduction and timing
non-overlap. All source snapshots/recursive inventories exclude the read-only
shared `third_party` junction and temporary files. The protected unrelated
untracked notes were not read, edited, or staged. No push was performed.
