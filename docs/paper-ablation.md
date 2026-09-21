# Paper ablation study

This experiment isolates three implementation themes: device residency and
synchronization, fusion and memory locality, and DCT arithmetic. It uses the
complete public encoding call and keeps effort policy, transform candidates,
AQ update count, final-score policy, and CPU thread count fixed within each
comparison. It does not change compression features to manufacture a speedup.

The code is on `experiment/paper-ablation`, based on
`b1a7373beff4c0370494c2f9584bcb1510491357`. Controls require the opt-in
`GJXL_BUILD_ABLATION_EXPERIMENT` build; ordinary builds ignore the environment
variable. They are private experiment infrastructure, not a supported encoder
API. Use a separate build directory.

## Comparisons

These are **conditional comparisons**, not a factorial experiment. The DCT
scalar implementation lacks the fused image and AC paths, so simply changing
the production DCT switch would confound arithmetic with dataflow.

| Factor | Optimized arm | Disabled arm | What changes |
|---|---|---|---|
| AQ synchronization | `production` | `aq-sync` | Split the same device policy into a prelude, each AQ evaluation/update, and final frame, with a completion wait at each boundary. Same kernels, buffers, update count, and final materialization. |
| ACS/AQ residency | `production` | `ac-handoff` | Keep GPU greedy selection, but restore its host handoff and host metadata preparation before AQ. This includes the change from device indirect dispatch to host-known dispatch. |
| Malta fusion/locality | `production` | `split-malta` | Replace the fused tiled Malta operation with its existing scale-to-global-scratch and response kernels. Both use the same scalar helper and reciprocal arithmetic. |
| EPF/conversion fusion | `production` | `split-epf` | Materialize filtered XYB, then convert to linear RGB separately. Keep the EPF implementation and pass count. |
| AC candidate fusion | `host-fused` | `host-split-ac` | Disable fused candidate forward/residual/inverse/loss stages. Both use the CPU greedy selector and SIMD DCT, with image DCT enabled in AQ. |
| DCT image locality | `host-split-ac` | `packed-simd` | Replace direct image DCT input/output with gather, packed SIMD DCT, and scatter. Both use split AC evaluation and the same host selector. |
| DCT arithmetic | `packed-simd` | `packed-scalar` | Change scalar versus SIMD matrix multiplication for the same seven transform shapes, on the same packed dataflow, split AC evaluation, and host selector. |

`host-fused` is a common reference arm, not a claim that CPU selection is the
production configuration. Do not attribute its gap from `production` to a
single kernel optimization. Likewise, `aq-sync` measures added synchronization
while buffers stay resident; it is not a GPU-versus-CPU AQ algorithm comparison.
On unified memory, mapped reads and explicit copies are host access operations,
not PCIe traffic. Their byte counters overlap and must not be added together.

The locality coverage here is direct-image DCT and tiled Malta, plus the
intermediate storage avoided by AC and EPF fusion. It does not disable all
Butteraugli tiling or every optimization in the encoder.

## Build and battery-friendly trial

Requires macOS/Metal, the repository's audited compiler, CMake, Python 3.11+,
`djxl`, and `ssimulacra2`. The trial accepts only its generated 128x96 smooth
and 257x193 textured/edge images, efforts 5 and 8, and distance 1. It uses two
CPU threads, one process per configuration/case, and one measured sample.

```sh
git submodule update --init third_party/metal-cpp
cmake -S . -B build-ablation -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_BUILD_ABLATION_EXPERIMENT=ON \
  -DGJXL_BUILD_FRONTIER_EXPERIMENT=OFF
cmake --build build-ablation --target gjxl_ablation_benchmark -j 2
python3 tools/ablation/run.py --output ablation-runs/trial \
  --decoder /absolute/path/to/djxl \
  --ssimulacra2 /absolute/path/to/ssimulacra2
```

The runner completes an incremental build before freezing its executable. It
records the base commit, hashes of source files (including experiment files),
a source archive and patch, CMake cache, submodule state, executable/tool/input
hashes, machine identity, power records, raw samples, decoded pixels, and
SSIMULACRA2 scores. Input loading, output writing, decoding, and quality scoring
are outside the measured call. Decoding explicitly requests linear sRGB.

Each process first prepares the encoder, performs the configured warmups, then
audits one additional untimed warm encode. Timed encodes disable the audit and
GPU stage profiling. All calls in a process must produce identical bytes.
The audit records actual kernel invocations, selector and handoff choices, AQ
evaluation counts, submissions, waits, and host access sizes. Indirect counts
are encoded dispatches, including possible zero-work grids; they are not a
measure of GPU work. The runner rejects unexpected paths and unknown kernels.

Scheduling and dataflow pairs require codestream identity. The DCT arithmetic
pair records byte identity, decoded RMSE/max error, size ratio, and the change
in SSIMULACRA2. Any DCT output difference is marked `requires-quality-review`,
not silently treated as a pure speed result. Identical tiny trial outputs do
not establish corpus-wide numerical equivalence.

## Resume and full collection

Use exactly the same command with `--resume`. A lock prevents simultaneous
writers. Per-job records are atomic; interrupted jobs are rerun. Completed
jobs are verified against retained artifact hashes and raw records, and the
runner refuses changes to the source, binary, tools, inputs, machine, or
protocol. `record.json` is the completion marker; `ledger.jsonl` is an
append-only event log. A crash between the marker and log may omit a log event;
the report is reconstructed from verified records.

Full collection requires explicit `--full`, explicit PFM inputs, a fresh output
directory, and AC power. It stops between jobs if AC power is lost. It uses
three process rounds, three warmups, seven timing samples, and eight CPU
threads; arm order rotates across cases/rounds. Pin the final patch or commit
and build before starting a corpus study.

```sh
python3 tools/ablation/run.py --full --output ablation-runs/corpus-v1 \
  --input /corpus/image-a.pfm --input /corpus/image-b.pfm \
  --efforts 5,8 --distances 0.5,1,2 \
  --decoder /absolute/path/to/pinned/djxl \
  --ssimulacra2 /absolute/path/to/ssimulacra2
```

The supported cohort is efforts 5–9. Efforts 1–4 omit mixed-transform selection
and iterative refinement; effort 10 has a separate dense candidate policy.
They need separate study definitions, rather than silently contributing
inactive controls to this experiment. Include representative photographs,
graphics, textures, odd dimensions, and image-size cohorts in the final corpus.

`REPORT.md` gives per-case paired ratios; `summary.json` and per-job `raw.json`
retain the underlying results. For paper results, aggregate paired ratios at
the image level and show variation across process rounds and size cohorts.
Do not add conditional speedups together. Review nonidentical DCT cases for
rate/quality changes before claiming equal-output or equal-quality speedups.
No run is automatically certified as publication-quality performance evidence.

## Validation

```sh
python3 tests/ablation_runner_test.py
cmake --build build-ablation --target gjxl_dct_test gjxl_metal_aq_evaluation_test -j 2
ctest --test-dir build-ablation \
  -R '^(ablation_runner|metal_dct|metal_aq_evaluation)$' --output-on-failure
GJXL_ABLATION_VARIANT=aq-sync build-ablation/gjxl_metal_aq_evaluation_test
```

The existing DCT test compares implementations with a numerical reference on
common input fixtures. The AQ test covers policy materialization, zero-update
cases, final-score behavior, ownership, and injected failures, including the
new segmentation path when invoked with `aq-sync`.

Retained local trial results and ordinary-build parity evidence are summarized
in [paper-ablation-trial.md](paper-ablation-trial.md). Battery trial timings
are only infrastructure checks; no optimization speedup is claimed from them.
