# Shared forward DCT bases for larger Metal AC strategies

Select shared forward bases for DCT32, 32×16 and 16×32. DCT8 keeps its existing
per-channel forward launches: grouping makes it roughly 16% slower even with
a shared basis. All existing shuffle tails remain enabled.

The selected padded-4K GPU stages change from 15.436 to
13.764 ms (-10.72% paired), with gains in all five pairs.
Whole-encode results are smaller and noisier, as detailed below.

Baseline: `ae2f2e46cfbcef46198f04480988ad2c2c27cfed` on `perf/metal-kernel-dataflow`.
Measured September 8, 2026 on Apple M4 Pro / Mac16,7, 20 GPU cores, 48 GiB,
macOS 15.6, Apple Clang 17, Release C++20/libc++. Results are incremental to
[the shuffle tails for the four staged strategies](metal-ac-shuffle-remaining.md).

## Implementation

The three selected shapes now place X/Y/B in one forward-DCT threadgroup.
Each basis element has one cooperative writer. The existing unconditional
helper barrier publishes both the immutable basis and channel-local pixels.
Each channel retains its original SIMD-group matrix order, scaling, pixel
gather and coefficient indices. Only the X-channel owner computes a requested
forward quant norm, as before.

Device coefficients remain between the forward and residual/inverse/loss
dispatches. The latter keeps its existing per-channel launches, tuned worker
counts, shuffle reductions and basis staging. The scalar cost finalizer and
three-dispatch stage boundary also remain. This experiment neither completes
candidate fusion nor reuses the forward basis in the inverse dispatch.

The host selects each optional grouped-forward pipeline independently on
Apple family 9 when fused forward and tuned loss are available. Selection
checks 32-lane SIMD width, maximum threads and threadgroup storage capacity.
Missing/unsupported pipelines retain the existing forward path. The old
shader library exercises this fallback in the selected variant baseline
confirmation runs. Existing complete fused candidates are unaffected.

## Resources and launches

Queried static threadgroup storage matches these values. Forward kernels have
no dynamic threadgroup allocations. The aggregate columns cover all three
channels of one candidate; the old per-group column is a single channel.

| Shape | Old bytes/group | Old bytes/candidate | Grouped private bytes | Grouped shared bytes | Old → grouped threads/group |
| --- | ---: | ---: | ---: | ---: | ---: |
| DCT8, rejected | 512 | 1,536 | 1,536 | 1,024 | 32 → 96 |
| DCT32 | 8,192 | 24,576 | 24,576 | 16,384 | 128 → 384 |
| 32×16 | 7,168 | 21,504 | 21,504 | 11,264 | 128 → 384 |
| 16×32 | 7,168 | 21,504 | 21,504 | 11,264 | 64 → 192 |

The forward grid changes from three groups per candidate to one; total
threads per candidate are unchanged. Individual groups are larger even though
aggregate storage falls. These are shader resource changes, not device-buffer
allocation or RSS savings. No new occupancy/limiter capture was taken.

## Three-arm screen

The original per-channel launch, grouped channels with private bases, and
grouped channels with one shared basis use the same benchmark executable.
A separate direct private-versus-shared comparison isolates basis sharing
under identical grouped launch geometry. Each comparison uses three
alternating process pairs at padded 1080p and padded 4K, with three warmups
and seven measured calls per process. Negative changes mean faster.

| Padded-4K GPU stage | Grouped private vs old | Grouped shared vs old | Shared vs private, direct |
| --- | ---: | ---: | ---: |
| DCT8 | +16.37% | +16.29% | -0.32% |
| DCT32 | +3.19% | -7.00% | -9.87% |
| 32×16 | +3.78% | -8.61% | -12.14% |
| 16×32 | +10.75% | -17.05% | -25.16% |

Grouping with private bases regresses every shape. Sharing offsets that cost
for the three larger shapes at both resolutions. It gives DCT8 little benefit
relative to private grouping and leaves the regression versus the old path.
The DCT8 grouped kernels and private-basis control remain experimental only.

## Selected GPU-stage confirmation

Five alternating process pairs per input, three warmups and seven samples.
Supported stage timestamps include the unchanged inverse/loss and finalizer
alongside the changed forward kernel. Individual dispatch times are unavailable
on this device. Only the intended forward entry points and grid/thread geometry
change; the recorded inverse/finalizer dispatches remain identical.

| Padded-4K stage | Baseline ms | Selected ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| DCT8 | 2.482 | 2.478 | -0.09% | 3/5 |
| DCT32 | 5.909 | 5.442 | -6.21% | 5/5 |
| 32×16 | 4.749 | 4.405 | -6.88% | 5/5 |
| 16×32 | 4.796 | 3.868 | -19.87% | 5/5 |
| Three selected stages combined | 15.436 | 13.764 | -10.72% | 5/5 |
| All AC stages | 29.456 | 27.582 | -5.81% | 5/5 |

| Input, three selected stages | Baseline ms | Selected ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 128×96 | 0.090 | 0.086 | -6.03% | 5/5 |
| Padded 1919×1079 | 3.645 | 3.346 | -8.77% | 5/5 |
| Padded 3839×2159 | 15.436 | 13.764 | -10.72% | 5/5 |
| Planter 4K | 15.192 | 13.944 | -9.47% | 5/5 |
| Doughnut 4672×5584 | 47.962 | 45.412 | -9.81% | 5/5 |

## Warm whole encoding

Seven alternating process pairs per input; this count was fixed before the
confirmation campaign. Each process uses three warmups and seven measured
calls. Every pair is retained, with stage and whole-call campaigns separate.
The two sides use the identical selected-host benchmark executable and differ
only in the explicit shader-library path. GPU profiling and Metal API/shader
validation are disabled for whole-call timing.

Boundary: `metal-public-workflow`, fully resident AQ, SIMD, fused-tuned AC,
distance 1.2, effort 7, default density/compression and no requested final score.
Input preparation, backend selection, quantization, codestream encoding and
summary assembly are included. File I/O, JPEG conversion and process/backend
startup are excluded. The doughnut PFM is byte-identical to the historical
4672×5584 input. Its earlier ~702 ms standalone measurement is context only;
the paired baseline below is the comparison for this experiment.

| Input | Baseline ms | Selected ms | Paired change | Faster pairs | Pair range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128×96 | 9.907 | 10.586 | +4.13% | 3/7 | -12.51% to +20.26% |
| Padded 1919×1079 | 63.370 | 63.151 | -0.01% | 4/7 | -4.58% to +5.44% |
| Padded 3839×2159 | 216.449 | 220.316 | -0.72% | 5/7 | -2.45% to +8.19% |
| Kodak 17 | 19.703 | 19.734 | -1.76% | 4/7 | -4.84% to +3.17% |
| Planter 1080p | 66.698 | 66.500 | -0.73% | 6/7 | -2.52% to +1.71% |
| Planter 4K | 205.129 | 205.410 | -0.26% | 4/7 | -0.96% to +0.86% |
| Doughnut 4672×5584 | 688.097 | 685.348 | -1.97% | 5/7 | -4.02% to +1.16% |

The clearest whole-call observation is Planter 1080p (-0.73%, 6/7 pairs).
Doughnut favors the candidate in 5/7 pairs (-1.97% paired median), but its
separately aggregated medians are 688.097 versus 685.348 ms. The paired result
should not be interpreted as a guaranteed 2% improvement on every encode.
Padded 4K favors the candidate in 5/7 pairs (-0.72%) but includes two positive
outliers; its separate candidate-time median is higher. Padded 1080p and
Planter 4K are effectively flat, and Kodak is mixed (4/7 pairs).

Synthetic 128×96 has a +4.13% whole-call paired median, only 3/7 faster pairs,
and a -12.51% to +20.26% range. Its selected GPU-stage total improves in 5/5
pairs, while whole-call input preparation and quantization vary substantially.
DCT32 alone is +3.83% at that tiny GPU-stage boundary. The positive whole-call
estimate is retained as a limitation; no tiny-image speedup is claimed.

The three selected changes are retained for consistent combined GPU-stage
savings and exact output parity. Whole-encode benefit is modest and workload
dependent. All originally planned timing pairs are retained.

Times are medians of process medians; changes are medians of paired relative
changes, so they need not equal ratios of displayed times. Combined GPU times
are summed within a sample before taking medians. Ranges are observed spread,
not confidence intervals. Stage improvements are not whole-encode speedups.
These measurements are specific to this device/compiler and warm workloads.

## Qualification and evidence

- Private and shared variants: each passes 1,200 guarded cases normally and
  1,200 under Metal API/shader validation (4,800 total).
- The selected three-shape library passes another 1,200 normally and 1,200
  under validation. The forward stage is compared before inverse processing;
  full staged results and final costs are also compared bitwise.
- Existing staged mode passes 1,200; the original split-oracle, fused-baseline
  and validated fused-baseline modes pass 900 each. Total: **11,100 cases**.
- Probe coverage retains four shapes, five batch counts (1/2/7/33/257), ten
  signal/error patterns, three quant-norm sources and two row paddings. All
  buffer guards, padding, coefficients, quant norms, rates, loss and costs match.
- Effective Release result: **123/124**. The inherited `quantization_pipeline`
  mismatch is reproduced independently on the baseline: index 1, actual
  `0.24919039011001587`, expected `0.24914586544036865`.
- The initial full run also flagged an obsolete profiling-test expectation
  requiring `_forward_fused`. That test now validates grouped or fallback
  names, forward geometry and the retained per-channel inverse geometry.
  Its complete 13-test CLI suite passes on rerun; both original and rerun logs
  are retained. The three focused Metal suites pass under API/shader validation.
- All **57** canonical/policy/doughnut encodes match byte for byte; **four**
  decoded pairs match with pinned `djxl` revision `e8ff0976`.
- The initial experimental rectangle wrappers used the small-rectangle scale
  by mistake. The guarded forward comparison caught it before timing. The
  corrected variants use the original large-rectangle scale; failed artifacts
  and their log remain under `preflight-scale-typo/`.

Five timing campaigns retain 184 processes and
1288 measured calls. Runtime/input/output hashes,
nonoverlapping command intervals, and normalized dispatch inventories are
audited. Sources, libraries, private/rejected controls, raw samples and exact
commands remain under `build/shared-dct-forward/`. `analysis.json` stores paired
results; `qualified.json` and `verification.json` record qualification/provenance.
`prepare.py`, `extend_probe.py`, `build_screen.py`, `screen.py`,
`qualify_selected.py`, `confirm.py`, `analyze.py`, and `verify.py` document
construction and checks. Use a new experiment directory for a fresh run.

Selected metallib SHA-256:
`b9cb6c302dbc7e377f597b1918e749ddc3a7beac0b9b29bf4bc8d7c17869e35c`.
Promoted production sources match the qualified source snapshot exactly.

To repeat warm paired timing with frozen artifacts, choose a new output directory:

```sh
python3 tools/metal_dataflow/compare.py \
  --baseline-build build/shared-dct-forward/selected-baseline \
  --candidate-build build/shared-dct-forward/selected \
  --workload padded_4k --pairs 7 --warmups 3 --samples 7 \
  --output build/shared-dct-forward-repeat/wall
```

Add `--profile` and use a separate output directory for GPU-stage timing.
