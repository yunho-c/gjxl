# Fine Metal Butteraugli profiling

Butteraugli phase profiling is opt-in. Normal encoding uses
`GpuProfilingMode::kDisabled` and ordinary compute submissions; it does not
construct the fine stage graph or add timestamp sampling and profiling encoder
boundaries. The benchmark enables profiling with `--gpu-profile stage` and
`--gpu-profile-output PATH` in a resident, Metal-only public workflow.
Capture ordinary wall-time samples separately from GPU profiles.

## Phase boundaries

Reference and distorted psycho-image construction expose eight stages per
scale, in dependency order: `opsin`, `low_medium`, `high_x`, `high_y`,
`medium_b`, `suppress_x`, `ultra_x`, and `ultra_y`. Opsin includes any required
expansion or subsampling; ultra Y includes the existing fused raw-mask producer.
Reference preparation adds a ninth `mask` stage for mask blur and erosion.

Distorted stage IDs use `butteraugli.psycho.main.*` and
`butteraugli.psycho.sub.*`; their group IDs retain the former coarse names.
Reference IDs use `frontend.prepare_aq.reference.main.*` and `.sub.*`, with
`frontend.prepare_aq.reference` as their submission and group ID. Consumers
must aggregate all reference stages instead of assuming one stage per submission.

Ordinary and profiled execution share phase bodies and preserve dispatch order.
The profile storage plans account for additional stages, contexts and longer
IDs. `tools/metal_dataflow/compare.py` reconstructs coarse aggregates after
summing the disjoint fine stages, avoiding double counting. Comparisons with an
older coarse profile use their common boundary; raw JSON retains fine phases.

## Recorded measurements and validation

On an M4 Pro (20 GPU cores, 48 GiB), macOS 15.6, Xcode 26.3, Release,
fully-resident Metal at distance 1.2 and effort 7, padded-4K low/medium filtering
accounts for about 15.3 ms across reference preparation and two AQ feedback
iterations. Ultra Y plus raw-mask production accounts for about 7.4 ms. These
are instrumented attribution measurements, not a decomposition of ordinary
wall time or proof of a particular hardware bottleneck.

The isolated coarse-versus-fine comparison holds rectangular AC fusion constant.
At 4K, instrumented GPU stage totals change -0.02%, while the resident command
buffer span changes +0.30% and its host operation span +0.36%. At 128x96, the
latter spans increase 11.60% and 12.14%. Fine instrumentation therefore has
meaningful overhead on tiny workloads. The unprofiled whole-call comparison
changes +0.06% at 4K and +0.62% at 128x96, with mixed pair directions; shared
host-code refactoring is not claimed to have literally zero cost.

Overhead stage measurements use three alternating process pairs, three warmups
and five samples per process. Overhead whole-call measurements use five
alternating pairs, three warmups and nine samples. Phase attribution uses a
separate three-pair, two-warmup, three-sample cohort. Changes are medians of paired
ratios. Evidence and runtime hashes remain under `build/kernel-tuning/evidence/`, in
`profile-overhead-stage/`, `profile-overhead-wall/` and `final-stage/`.

The profiling implementation matches the qualified final source snapshot.
Recorded Release suites passed 121/122 tests for both the parent and final
candidate, with only the inherited CPU `quantization_pipeline` golden mismatch.
The final candidate included the now-committed rectangular AC fusion as well as
this profiling change. All 56 corpus/policy encode pairs were byte-identical;
seven selected ASan/UBSan tests and eight selected Metal validation tests passed.
Host sanitizer runs used the existing metal-cpp suppression and disabled leak
detection. Details remain in `qualification-final/`, `final-parity/` and
`asan-tests.log` under the evidence directory.

Profiling-specific tests require exact equality of ordinary and profiled AQ
scores, quantization fields and block distances. CLI tests check phase order,
dispatch counts and group IDs; storage tests cover expanded, single-scale and
multiscale images; comparison tests reject mixed coarse/fine inventories and
check that compatibility aggregates do not inflate GPU totals.
