# Small-image DC context trees

Complete-frame encoding now defaults to **weighted DC prediction and
size-adaptive predefined DC trees**, following the approved combined-policy
review on 2026-09-13. This ports pinned libjxl's `MakeFixedTree` sample-count
pruning to GJXL's fixed DC subtree. Weighted applies at every effort and image
size; the tree shape alone adapts to size. Explicit gradient prediction remains
available. Ordinary DC rounding and disabled adaptive DC smoothing are unchanged.

The original no-size-growth gate failed. Promotion deliberately accepts the
measured size/latency tradeoffs; it does not reinterpret that gate as passing.
The internal `ScopedDcTreePolicyForTesting` selects either layout for paired
qualification; it is not a public API, CLI flag, or environment switch.

The sample count is `3 * ceil(width / 8) * ceil(height / 8)` for the whole frame.
The selected number of DC leaves is 1 for 1–512 samples, 2 for 513–4096, 4 for
4097–8192, and 34 thereafter. These are the distinct outcomes of libjxl's
`min_gap = 8 * (14 - ceil(log2(samples)))` rule below log 14. For example,
257×193 uses two leaves; Kodak 768×512 and 4K use the full tree.

Only the DC subtree changes. Stream routing and the eleven AC-metadata leaves
retain their split and prediction behavior. Pruning changes breadth-first leaf
numbers, so a compile-time map translates all 45 legacy context IDs, including
metadata. One immutable layout supplies wire tokens, context counts, histogram
construction, candidate size measurement, and final emission. Full layouts skip
remapping and retain the legacy wire tokens exactly. Existing storage bounds
remain conservative: the layouts are static, the token scratch is still bounded
by 313 entries, and there is no new owned heap scratch. Geometry-less helper APIs
continue to use the full tree.

`tests/dc_context_tree_test.cpp` checks transitions, geometry, invalid inputs,
metadata mapping, extreme DC properties, and legacy weighted tokens. Its optional
oracle target compares the actual pinned libjxl predefined DC tree and decodes
GJXL's entropy-serialized tree with libjxl. Serializer and workflow budget tests
also exercise the adaptive policy, including prediction-aware quantization and
smoothing, repeat/profile equivalence, joined workers, and atomic failures.

## Combined-default decision

The promotion compares weighted/adaptive with gradient/legacy, using the retained
cohort's 216 image/effort/distance points. All cross-predictor decoded float hashes
match. Aggregate file-size changes (negative means smaller) are:

| Stratum | Points | Aggregate size change | Larger points |
|---|---:|---:|---:|
| Eight compact controls | 72 | -0.4577% | 13 |
| Twelve photographic thumbnails | 108 | +0.0501% | 65 |
| Four 4K photographs | 36 | -0.9865% | 0 |

This is a balanced default, not a per-image win. The thumbnail aggregate is
nearly neutral; the worst combined increase is nine bytes (+1.4658%) on the
edge control at e3/d0.6. The broader [predictor study](../dc-prediction-qualification/REPORT.md)
found photographic savings with roughly 3–4.5 ms added complete encoding time
and 2–3 ms added decoding time for its timed 4K cases. Tree pruning does not
remove that weighted-predictor cost. These are same-decoded-pixel writer results,
not a matched-quality comparison with stock libjxl.

C++, C, Rust, and both encoding CLIs share the new defaults. Older C structs
without a predictor field also select weighted; explicit gradient enum values
are unchanged. Geometry-less low-level helpers keep the legacy full-tree
contract. The old gradient/full-tree byte fixtures remain explicit regression
checks, and workflow resource tests cover the adaptive default and legacy trees.
Frozen qualification sources and exported `legacy_default` flags describe the
original study and are intentionally preserved.

## Default-promotion validation

`build/dc-weighted-default-validation-v1` retains a separate correctness-only
check of the actual default entry points. Twelve quality-benchmark encodes
(compact, two thumbnail sizes, and 4K at e3/e4/e7) and six CPU/Metal CLI encodes
match the previously qualified explicit weighted/adaptive bytes exactly.
The pinned decoder reproduces their retained finite float-pixel hashes.
Three replacement CLI byte fixtures (balanced, maximum compression, and maximum
error) also decode identically to their original gradient/full-tree fixtures.
The portable [default-validation.json](default-validation.json) records all
21 checks and binary hashes. No new performance conclusion is drawn from these
correctness runs.

The Release suite plus focused reruns passes 144/145 tests. The only remaining
failure is the previously reproduced `quantization_pipeline` golden documented
below. The fresh Rust/native build passes 10/10; focused ASan/UBSan tests pass
4/4. Default serializer tests cover all four tree shapes, and C/C++/Rust/CLI
tests cover explicit gradient selection. Legacy byte fixtures stay explicit;
target-byte success fixtures use smaller reachable budgets with the same
tolerance and attempt limits. Complete-workflow profiling now forwards the
serializer's per-frame DC sample, leaf, and context counts.

Logs are `build/release/dc-default-tests.log`,
`build/release/dc-default-workflow-final.log`,
`build/release/dc-default-cli-final.log`,
`build/release/dc-default-final-checks.log`,
`build/dc-default-rust-tests-final.log`, and
`build/sanitize/dc-default-tests.log`. The first full run and intermediate logs
retain the failures corrected during promotion; the final workflow and CLI
reruns supersede those failures.

## Qualification protocol

The frozen study starts from `6ca9d5f`. It uses eight retained compact controls,
256- and 384-pixel thumbnails from first/middle/last sorted CLIC and Kodak images,
and four retained 4K photographic controls. Each is encoded at e3/e4/e7 and
distances 0.6/1/2 using both predictors and both tree policies (864 streams).
Thumbnails use deterministic area averaging in linear RGB; source and derived
input hashes are retained. Additional CPU/Metal integration pairs cover round,
prediction-aware quantization, smoothing, and their combination.

Timing uses edge, the first CLIC 256 thumbnail, and 4K planter, at e3/e4/e7 with
both predictors. Each process makes three warmups per policy and twenty AB/BA
pairs. The interval is the complete encode call; decode is measured separately
with a fresh single-thread decoder, including decoder creation/destruction and
output allocation (the returned pixel buffer is freed outside the interval). Validation
runs outside timing. The retained decoder helper calls its first/second inputs
`gradient`/`weighted`; this study explicitly maps those labels to legacy/adaptive.
No quality calibration is needed for an exactly decoded-pixel-identical writer
change.

The gate requires all correctness/resource checks, exact full-tree bytes, no
qualification case growth, aggregate savings on affected images for each
predictor, and no repeatable regression larger than max(0.1 ms, 2%) with at least
16/20 slower pairs. An offending case is repeated once under the same quiet
checks. Rejected/interrupted attempts retain their raw artifacts. The cohort and
thresholds must not be tuned in response to results.

The source-bound harness is `tools/dc_coding/tree_study.py`. `init` freezes sources,
tools, build commands, inputs, and the cohort. `collect` and `timing` are explicit,
resumable commands; `--max-cases` bounds either. `report` only reads saved artifacts
and writes size/integration/timing CSVs plus a completion/gate summary. Use a new
output namespace after any source or binary change. Tests and builds must finish
before performance collection. Process and power snapshots accompany timing;
a busy attempt is retained but not accepted. Collection rejects another codec
measurement/build or a non-kernel process above 100% CPU. Timing additionally
rejects non-kernel processes above 80% CPU before and after each case. These are
process snapshots, not a guarantee of an otherwise idle operating system.

## Initial qualification result

The initial qualification retained the legacy default. The complete size cohort
passed exact pixel equality, but the size gate failed: 146/432 predictor comparisons grew
(52 gradient, 94 weighted). Among the 180 affected comparisons per predictor,
gradient saved 3,669 bytes in aggregate (−0.1477%), while weighted added 1,613
bytes (+0.0650%). Weighted photographic thumbnails alone grew by 4,568 bytes
(+0.2316%). The largest individual increase was 65 bytes (+1.9884%) for the first
CLIC 256 thumbnail at e4/d2 with weighted prediction. Flat inputs saved as much
as 43.2%, which does not offset the predeclared no-growth requirement.

All 432 cohort pairs and all 48 additional CPU/Metal integration pairs decoded
to exactly equal finite float pixels (960 retained codestreams). All 72 cohort
pairs that select the full tree were byte-identical. The integration matrix also
had 16 size increases. These are same-input, same-encoder-control writer results;
they do not compare GJXL quality or speed against libjxl.

Release verification passed 144/145 tests after restoring the pinned testdata
submodule. The sole failure is the inherited `quantization_pipeline` golden:
actual 0.24919039011001587 versus expected 0.24914586544036865. Focused native
ASan/UBSan checks passed 4/4. Leak detection is unsupported by this macOS ASan
runtime; the successful run used `detect_leaks=0`. The external pinned archive is
not sanitizer-instrumented, and its oracle adapter omits vptr checking because
the archive was compiled without RTTI.

The accepted collection lives at
`build/dc-small-tree-qualification-v2`, with `manifest.json`, `source.zip`,
`source.diff`, all JXL files, independent decoder pixel hashes, commands, and
validation logs. Version 1 contains only an unused input-generation checkpoint;
it was superseded before collection to replace macOS BLAS matrix products with
explicit sparse area sums. Failed launches and quiet-guard rejections are retained.

The adjacent `sizes.csv`, `integration.csv`, `timing.csv`, and `summary.json` are
saved-data exports. `summary.json` explicitly records completion and the failed
measurement gate. The source freeze remains the legacy-default implementation;
no threshold or cohort tuning was performed after inspecting the results.

## Latency qualification

All 18 cases completed with 3 warmups per variant and 20 alternating pairs;
no case met the joint size-of-regression and 16/20 slower-pairs threshold, so
no confirmation repeat was triggered. Negative values below mean adaptive
was faster. Values are the median of paired differences in milliseconds,
which need not equal the difference between the two independent medians.

| Input | Effort | Encode Δ gradient (ms) | Encode Δ weighted (ms) | Decode Δ gradient (ms) | Decode Δ weighted (ms) |
|---|---:|---:|---:|---:|---:|
| CLIC thumbnail 256 | 3 | -0.025 | -0.063 | -0.007 | -0.023 |
| CLIC thumbnail 256 | 4 | -0.195 | -0.288 | -0.009 | +0.001 |
| CLIC thumbnail 256 | 7 | +0.054 | +0.079 | -0.033 | -0.001 |
| Edge 257×193 | 3 | -0.106 | -0.048 | +0.000 | -0.002 |
| Edge 257×193 | 4 | -0.081 | -0.175 | -0.003 | -0.012 |
| Edge 257×193 | 7 | +0.029 | -0.093 | -0.012 | -0.004 |
| Planter 4K | 3 | +1.401 | +1.436 | -1.030 | -0.236 |
| Planter 4K | 4 | -0.622 | -0.928 | +0.722 | +0.169 |
| Planter 4K | 7 | -0.824 | -0.712 | -2.183 | +3.518 |

The 4K controls use the identical full tree and identical stream bytes. Their
variation (including +3.518 ms for weighted e7 decode, with 15/20 slower
pairs) illustrates residual machine noise despite the guards. These
measurements do not establish a speed gain. Warmup settings,
all twenty accepted pairs, and excluded attempts remain auditable in the raw
directory; `exclusions.csv` lists rejected launches and unaccepted runs.
