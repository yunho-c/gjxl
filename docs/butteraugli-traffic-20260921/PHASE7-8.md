# Butteraugli traffic study: phase 7-8 checkpoint

> Historical study checkpoint. The final disposition and commit status are in [REPORT.md](REPORT.md) and [README.md](README.md). Any pending work below refers to the time of that checkpoint.

The currently supported candidate remains `integrated/fusion-bundle`: exact AC/DC producer composition, two-axis/two-output low/medium reuse, and direct 7-tap ultra filtering. Its broad three-pair ordinary cohort improves large-image latency 5.0-7.4%; seven-pair confirmation is running. This checkpoint supersedes the broader-correctness pending statements in `PHASE6.md`.

## Broader correctness of the measured bundle

The live study worktree was restored to the exact frozen bundle sources. The rebuilt shader library is byte-identical to the measured library. Seven Metal/resident tests pass under API/shader validation, including operation contracts, AQ policy, resident storage/legacy DC-tree paths, and cache admission. Fifty-six canonical/policy CLI comparisons preserve output bytes; the pinned decoder produces three identical decoded pairs, independently checked for dimensions, payload length and finite pixels with Python standard-library arrays.

See `phase8-closure-r1/bundle-expanded-contracts.log`, `fusion-bundle-canonical-parity/summary.json`, `fusion-bundle-canonical-parity/finite-pixels.json`, and the source/binary hashes in `integrated/fusion-bundle/identity.json`. Full-suite success is not claimed: the previously documented unrelated optional C API build failure and CPU golden failure have not been repaired or waived.

## Mask-producer composition: rejected

Two variants fuse main-distance composition into the final existing distorted-mask blur pass. Both add no dispatch, replace a store, and preserve the original arithmetic. Each passes 192 guarded producer comparisons, 360 guarded reduction/error cases and three focused integration tests. Ordinary/profiling probes preserve paired output bytes.

| Orientation / affected scope versus AC/DC fusion | Alpine24/e7 | Forest48/e10 |
|---|---:|---:|
| Row-major: resident reduction | -65.459% | -72.085% |
| Row-major: main mask stage | +109.127% | +128.909% |
| **Row-major: mask + reduction together** | **+7.178%** | **+5.997%** |
| Transposed: resident reduction | -66.042% | -71.700% |
| Transposed: main mask stage | +116.604% | +126.786% |
| **Transposed: mask + reduction together** | **+8.807%** | **+5.664%** |

Three-pair stage measurements show that the faster reducer does not compensate for the more expensive mask producer. Reject both variants. These results are separate from the still worse materialized-map-plus-reduction experiment; they close an additional plausible producer-fusion mechanism, not every possible future design.

## Four-output refinements and final filter interactions

Against the two-output bundle, 16x64/four-output filtering improves the integrated low/medium scope 13.387% at 24 MP but only 0.553% at 48 MP. The 16x96/four-output alternative improves it 8.833% and 2.646%. Both have mixed instrumented whole-call results. Ordinary incremental comparisons are running; isolated microtiming is insufficient to select either over the supported bundle.

The final interaction screen repeats the two unbounded four-output configurations as within-cohort references, then tests two/four rolling stripes and actual-thread launch bounds. All eight configurations pass 48 guarded cases. Five-pair padded4K/24MP microtiming retains the original-kernel control (-0.05%/-0.06%). Explicit 256/384-thread bounds reproduce their unbounded counterparts to approximately 0.15 percentage points or better. Rolling stripes can help padded4K, but none surpasses the best single-stripe configuration across both extents; some substantially regress at 24 MP. This gives no consistent large-image reason to add rolling or launch-bound complexity to the finalists.

The initial closure compile used the function attribute after the parameter list and failed. The attribute was moved before `kernel void`; the failed source and diagnostic remain in `phase8-closure/`, and the successful retry in `phase8-closure-r1/`. No failed-build measurement was used. See `INCIDENTS.md`.

## Remaining bounded audit work

Campaign 9 performs ordinary incremental checks of the four-output/Malta refinements, seven-pair bundle/original confirmation on two large and two Kodak cases, seven-pair unchanged-binary controls on the large cases, and separate original/bundle stage attribution. A source/history audit also identified two remaining geometry leads: the 5-tap Opsin default was chosen partly for smaller-image balance, and the 13-tap mask transpose retains its original launch. Campaign 10 tests the previously studied 32x8 Opsin and 16x16 mask-transpose choices on the current large-image bundle, with separate stage and ordinary comparisons. These open checks prevent an exhaustion conclusion today.
