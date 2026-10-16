# Removing redundant histogram fitter work

2026-10-15. This change reduces CPU work inside `LearnHistogramTree`, for both
shared and local trees. It combines property occupancy with the first
predictor's joint histogram and carries constant-property information from a
parent to its children. Sampling, quantization, predictor policy, entropy
estimates, candidate order, threshold ties and partition order are unchanged.

On the three 12 MP photographs, this removes **41.8–42.4% of property/histogram
row passes** and reduces complete public CPU elapsed time by **10.9–13.1%** at
14 participants. The main capture's **146 complete encodes** retain exact bytes
and independently decoded samples. This is an implementation speed improvement
at the same rate and reconstruction.

The baseline is `5ae11e4`, the current `feat/modular` merge. Its fitter and
learning scheduler are identical to `a7ceac3`. The candidate is the uncommitted
source snapshot retained with this report's measurements.

The subsequent [continuous-scheduler integration](modular-fitting-work-reduction-integration.md)
checks this work reduction with the newer fitting queues and feature cohorts.
The measurements below retain the original wave scheduler.

## Why this work can be removed

Previously, every considered node/property visited its rows once to collect
unweighted occupancy and again for each predictor to collect weighted
property/residual histograms. It also updated the occupied minimum and maximum
bin on every row. This included properties with no possible split: constant
properties still collected every predictor's histogram, and descendants
rediscovered constants already established by their parent.

The implementation in
[`histogram_tree_learning.cpp`](../src/codec/modular/histogram_tree_learning.cpp)
now performs three reductions:

1. The first predictor's histogram pass also counts distinct rows per property
   bin. Those counts remain **unweighted** because they determine in-place
   partition positions; residual histograms and extra-bit totals retain sample
   weights. Occupied endpoints are found on the bin axis after accumulation.
2. When that pass discovers a constant property, it clears the first
   predictor's increments and skips the remaining predictors and scoring.
3. A bit mask in each pending node records constant property columns. Children
   inherit the parent's mask by value, so constants need no further row scans.
   A property with one quantization bin can be skipped immediately. Constants
   discovered inside one child do not affect its sibling.

A constant column has no candidate threshold and remains constant in every
subset of its rows. All nonconstant candidates are evaluated in their original
property/predictor/bin order with the original floating-point operations.
Forced multiplier splits are resolved before this ordinary property search,
including forced boundaries with an empty child.

The existing histogram, occupancy and cost buffers are reused. On this arm64
build, the mask increases `NodeInfo` from 48 to 56 bytes: at most **16,376 bytes
per fit** at depth capacity 2,047. The capacity planner already uses
`sizeof(NodeInfo)`, so both direct fitting and parallel job admission include
the increase. There are no new allocations or allocations after partitioning
begins.

This is a bounded first implementation of “reduce work inside the fitter.”
Full parent/child histogram subtraction and GPU histogram/scoring batches remain
separate work. Retaining joint histograms would add property × predictor × bin
× symbol storage for cached nodes; this change obtains reuse with one mask and
leaves numerical scoring and execution scheduling intact.

## Diagnostic meaning

The histogram diagnostic JSON is schema **2**; the attribution reader accepts
both historical schema 1 and schema 2. Logical property counts, constant counts,
occupied-bin distributions, candidate counts, entropy evaluations and row
swaps keep their meanings. The two new work counters are:

- `fused_property_row_visits`: occupancy visits shared with the first predictor.
- `skipped_constant_properties`: considered properties resolved from an
  inherited constant or a one-bin quantizer, without reading rows.

Actual property/histogram row passes are
`property_row_visits + joint_histogram_row_visits - fused_property_row_visits`.
Schema 1 has no fused visits. This excludes the unchanged symbol-range,
parent-histogram and partition passes.

`joint_histograms` now includes the fused occupancy work.
`property_population` covers occupancy-array clearing, bin endpoint lookup and
cost-array initialization. Its time decrease alone is not the work reduction;
compare the combined phases and physical row-pass counters. Detailed phase
times in a parallel encode are **sums of overlapping fit wall intervals**, not
an additive subdivision of elapsed training time.

## Validation

Release, Apple Clang 21, `-O3 -g -DNDEBUG`, Metal enabled; independent decoder
and fitting oracle at libjxl `e8ff09762481785938d8e4e01333ed3917571161`:

- Seven targeted root-build CTests passed: histogram learning/preparation,
  parallel learning, reference workflow, hybrid workflow, hybrid palette and
  public API. These include tree/byte parity, Best/All predictors, prefix/ANS,
  bounded memory and CPU participation, partial launches, resource release,
  admission failures and atomic publication.
- The native contract and **257 generated reference differential cases** passed.
  The added cases exercise constant first/last bins and properties that become
  constant in only one subtree, alongside predictor ties and forced multiplier
  boundaries, including empty sides.
- New native checks cover sparse dynamic bins through 255, static bins through
  65,535, saturated 16-bit weights, one/two/all 14 predictors, retained row order,
  changing symbol spans, histogram clearing and constant propagation.
- Native learning and preparation passed ASan/UBSan in the existing Debug
  sanitizer build with `ASAN_OPTIONS=detect_leaks=0`. This is focused CPU
  sanitizer coverage, not a whole-encoder or Metal sanitizer claim.
- The attribution reader validates all 20 historical schema-1 profiles; a
  separate two-encode schema-2 smoke capture passes the runner and report
  generator, including the revised phase labels. These checks are outside the
  main performance capture.

## Measurements

Apple M4 Pro, 14 CPU cores, 48 GiB, macOS 27.0.1, AC power observed before and
after every process. Public learned policy, Best predictors, estimated ANS,
14 CPU participants, automatic tree routing. CPU and explicit Metal use the
existing public wrapper; Metal retains automatic feature-producer selection.
Detailed fitter probes are disabled for the public timing comparison.

Each input/binary runs in a fresh process: one cold call and one warmup per
route, followed by **three measured rounds** with rotating CPU/Metal order.
Baseline/candidate process order reverses on alternate inputs. The encoder
timer includes admission, selection, preparation, fitting, tokens, modeling,
emission, cleanup and publication; input/output file I/O, domain construction
and independent decoding are outside it. Both executables use the same build
configuration and dependencies.

Seconds below are per-input medians. Percentages mean **less elapsed time**,
not speedup factors. The first five rows reproduce the earlier quick
qualification's input set; 1039 and 1049 extend its photographic coverage.

| Input | CPU before | CPU after | Reduction | Metal before | Metal after | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| photo-macan | 0.4625 | 0.4614 | 0.2% | 0.4529 | 0.4066 | 10.2% |
| alpha-ria | 0.6461 | 0.5334 | 17.5% | 0.6345 | 0.5186 | 18.3% |
| palette-358 | 0.0104 | 0.0093 | 9.9% | 0.0106 | 0.0092 | 13.1% |
| alpha16-ramp | 0.3420 | 0.2582 | 24.5% | 0.3357 | 0.2538 | 24.4% |
| photo-large-1029 | 5.2274 | 4.6591 | 10.9% | 5.7494 | 5.3803 | 6.4% |
| photo-large-1039 | 4.2061 | 3.7236 | 11.5% | 5.1385 | 4.5444 | 11.6% |
| photo-large-1049 | 3.9172 | 3.4051 | 13.1% | 4.3034 | 3.6664 | 14.8% |

Sums of the original five per-image medians decrease from **6.688 to 5.921 s**
on CPU (11.5%) and **7.183 to 6.568 s** on Metal (8.6%). Across the three large
photographs, the corresponding reductions are 11.7% and 10.5%.

For 1029, CPU elapsed training decreases from **3.426 to 2.866 s** (16.3%).
Median selection/token/parallel-stream-encoding intervals are respectively
0.415/0.202/1.141 s before and 0.421/0.211/1.137 s after. These separately
computed medians need not sum to the median complete call. The Metal training
interval decreases from 4.058 to 3.723 s. This change does not diagnose the
remaining CPU/Metal scheduling difference.

Desktop services remained active. There is visible variation: 1029 CPU ranges
are 5.205–5.318 s before and 4.656–4.935 s after; Metal ranges are 5.720–5.856 s
and 5.059–5.618 s. `photo-macan` CPU ranges are 0.461–0.468 s and 0.414–0.475 s,
so its near-zero median improvement should not be hidden behind the aggregate.
These three-repetition, seven-image results are a bounded qualification, not a
general corpus claim or an isolation of desktop interference.

## Work attribution on the parallel CPU path

One additional instrumented CPU encode per binary/large input records the
fitter's work and phases at 14 participants. These six diagnostic calls are
separate from the repeated public timing comparison. All node counts, splits,
forced splits, logical properties/constants, bin distributions, candidate
thresholds, entropy evaluations, symbol terms, row swaps, fit shapes and
complete output hashes match before/after.

| Image | Property/histogram row passes before / after (billions) | Pass reduction | Fitter work before / after (s) |
|---|---:|---:|---:|
| 1029 | 13.164 / 7.658 | 41.8% | 32.919 / 25.719 |
| 1039 | 10.637 / 6.147 | 42.2% | 22.784 / 16.600 |
| 1049 | 10.057 / 5.796 | 42.4% | 20.546 / 15.024 |

Fitter work here is a **sum of overlapping job wall times**, down 21.9–27.1%
in these diagnostic pairs. It is not the elapsed training interval or process
CPU time. Occupancy plus joint-histogram phase sums decrease from
18.340/15.289/13.826 s to 11.274/9.303/8.367 s. Candidate evaluation and partition
work remain substantial. The new implementation skips 562,149 / 295,704 /
250,757 already-known constant properties in these images.

## Retained evidence and reproduction

The [results ledger](modular-fitter-work-reduction-results.json) retains all
140 public calls (including 84 measured calls and 56 cold/warmup controls), six
diagnostic calls, per-call elapsed/process-CPU/stage times, resource peaks,
per-fit diagnostics, boundary environment snapshots, hashes and validation
logs. Every call independently decodes exactly; every route/repetition and
before/after pair has identical complete bytes. Learning, token and entropy
work checks also pass. The two exploratory pilot calls are retained separately
in the artifact directory and do not contribute to these tables.

`build/fitter-work-reduction-20261015` contains baseline/candidate executables,
baseline source archive, candidate source archive and patch, build flags,
inputs by referenced path/hash, all complete outputs, stdout/stderr, main
manifest, comparison/analysis scripts and test logs. Source, executable, input
and non-system dynamic-library hashes were verified at both capture boundaries.
Documentation and the Python report-label update followed capture; the C++
implementation and binaries are the measured versions.

Frozen public executable SHA-256:

- Before: `7f93f959ec425cfaf22c1b929873f30a38ea72a942864770c38807d9c7dd33e1`.
- After: `0296203c882dd5b8a847f1a632a8e09f693426c2e2aeb0b093f9afee3b62c82d`.

Build the same targets at the baseline and candidate snapshots, retaining both
binaries, then run each input with:

```sh
gjxl_modular_metal_workflow_profile INPUT.pam FRESH_OUTPUT 3 encode-public 14
```

For the separate parallel CPU work diagnostic:

```sh
gjxl_modular_hybrid_profile INPUT.pam FRESH_OUTPUT cpu auto best fit-on 14 14 14 14 14
```

The retained `compare.py` rotates binary order and enforces exact-byte, decoder,
resource-release, power and artifact checks. `analyze.py` verifies the
unchanged logical work and computes the linked report from its manifest.
