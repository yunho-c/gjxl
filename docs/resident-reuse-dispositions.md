# Resident reuse and fusion dispositions

This closes milestone 5's bounded inventory at `be00797`. It does not declare
all possible GPU optimizations exhausted. The required ownership/reuse changes
are already implemented; this audit identifies what remains intentionally
separate and why. No new kernel, alias, arithmetic change or policy pruning is
introduced by this checkpoint. Aggregate CPU scheduling remains milestone 6.

Later handoff experiments against `32cd437` are recorded in
[borrowed AC scores and adjusted quant fields](resident-borrowed-handoff-experiments.md).
They remove real copies and host buffers but have no established independent
whole-encode benefit; that follow-up retains evidence and revisit criteria.
The separately integrated packed-input change is documented in
[packed resident input](packed-resident-input.md).

## Inventory and decisions

The following producers, consumers and lifetime boundaries are the audited set.
Their capacities remain governed by the plans linked from
[resident resources](resident-resources.md), including active and idle storage.

| Storage/calculation | Producer, consumers and last use | Disposition |
| --- | --- | --- |
| Caller RGB, resident input and host mask | Caller view is borrowed; resident preparation produces padded opsin and quantization inputs. AC/AQ consume them. | **Implemented:** omit unrequested host masks, borrow resident planes in AC search, and release final-use prepared input before the independent-output CPU tail. C packed-to-linear conversion remains necessary for the canonical linear input contract. |
| AC gathered pixels and forward/residual temporaries | Per-family candidate kernels produce packed forward coefficients, then residual pixels/rates; the three-channel cost reduction consumes them. | **Implemented:** fused gather/forward and residual/inverse paths, two reused maximum-size scratch buffers, and final-search release after GPU completion and CPU placement. Keep the remaining inter-stage boundaries; see the dependency analysis below. |
| AC quantization norm | Fused forward computes a larger-block aggregate once per candidate in the temporarily unused cost output; residual and final cost kernels consume it before final cost replaces it. | **Implemented:** exact shared calculation for covered-block count at least four. Smaller candidates retain direct field evaluation. Do not add a separate permanent norm buffer or pretend this calculation is still repeated for every channel. |
| AC candidate/matrix metadata and scalar costs | CPU enumeration/matrix packing feeds each search; GPU scalar costs return for deterministic CPU placement. Reusable search state retains capacity through retries. | **Retain capacity reuse; defer additional content caching.** Candidate quantization/CfL inputs can change with an attempt, while matrix contents are immutable. Matrix-only caching would affect a subset of the measured 0.31/3.38 ms setup span, not the much larger search GPU work. Cross-image prepared-state caching would need a separate content/geometry validity contract; volatile backing caching deliberately does not promise that. |
| AQ forward coefficients | First reconstruction produces coefficients for the selected strategy; later quantizer/coefficient passes and final output consume them. | **Implemented:** preserve forward coefficients across iterations and final emission. The fresh trace confirms one forward pass per selected family, not one per iteration. Different quantization fields still require new coefficient decisions/reconstruction. |
| AQ filters/gather scratch versus Butteraugli | Filtering finishes into distinct linear RGB before metric work borrows two three-channel filter images; gathered transform planes are no longer needed once forward coefficients exist. | **Implemented:** nine transient shared planes, reduced half-resolution final staging, and reconstruction staging reused for complete-map fallback. Next-iteration producers overwrite scratch before reading it; borrower destruction precedes returning AQ arenas. Reference and retained coefficients are not aliased. |
| Butteraugli reference and reconstructed candidate | Reference preparation depends on the original image; each comparison depends on newly reconstructed/filter-processed pixels. Reference preparation happens once per prepared evaluator. | **Retain separate semantic state.** Cache owning capacity, not image-specific reference contents. Reusing reference contents across mutable caller inputs would need an immutable-image/content-identity API. Fusing neighborhood filters or multiscale metric work requires its own halo, reduction-order and numerical qualification. |
| AQ policy iterations and final emission | GPU reconstruction, filtering, metric reduction and policy update form an ordered dependency chain. Final emission uses the selected quantizer/field. | **Implemented:** the resident policy sequence is recorded in one AQ submission; it does not introduce a CPU round trip per iteration. Keep required iteration dependencies. Removing evaluations or predicting updates is the separate policy track, not a storage refactor. |
| Completed AC coefficients and block metadata | Final coefficient kernels write independent serializer destinations; a small host snapshot supplies block metadata. CPU frame views consume both after evaluator destruction. | **Implemented:** direct final destinations and ownership-independent views. **Intentionally retain** the independent metadata and output owner; tying them to a large AQ arena would extend its lifetime. Intermediate float coefficients and final packed integer groups have different layouts and consumers; in-place reuse needs an overlap-safe layout proof, not a pointer cast. |
| Serializer tokens, models, candidate writers and byte output | CPU stages consume the completed view, preserve candidate/tie policy, and retain winning/competing state until selection/publication. | **Implemented:** managed owners, bounded replacement overlap, direct token views, prepared entropy values, and existing equivalent-symbol-code sharing. **Defer new cross-candidate model/writer caches:** populations/configurations and traversal state are not interchangeable; a new executor-local cache would add another lifetime/budget policy before CPU scheduling is settled. Exact entropy algorithm tuning remains a separately measured optimization, not an unbounded requirement of this milestone. |
| C++/C and batch result publication | C++ transfers ordinary-vector backing; C copies into its byte-array ownership contract; batch retains every result until atomic array publication. | **Intentionally retain** the C copy and batch result envelope. A portable vector-to-`delete[]` ownership transfer is not valid; eliminating that copy needs a different backing/free contract. Streaming/spilling batch results would change the public API. These costs remain admitted. |
| Idle caches and completed-job capacity | AQ/input/Butteraugli owners return reusable capacity; later same-domain work may acquire it, or admission/trim frees it. | **Implemented:** accounted volatile caches, all-pool/domain-aware eviction, and tight-batch retirement. **Defer transactional idle-credit admission:** current complete reservations can evict otherwise reusable cache capacity under a tight cap. Avoiding that needs atomic ownership/credit transfer before admission, not removal of the hard-limit check. Its measured cost is explicit below. |

## Why the remaining AC boundaries are real dependencies

In [the AC kernels](../src/gpu/metal/kernels/ac_strategy.metal), transform groups
are indexed per candidate **and channel**. Residual/CfL work can consume the Y
forward coefficients while computing another channel. The final cost group is
indexed per candidate and reduces residual/rate values from **all three**
channel groups, with a specific reduction and channel-weighting order.

The dispatch boundaries in
[`EncodeAcStrategyCandidateBatch`](../src/gpu/metal/metal_ac_strategy.cpp)
therefore supply inter-group ordering. A threadgroup barrier inside the existing
per-channel kernel cannot replace them. Further forward/residual/cost fusion
would need a different multi-channel group decomposition, local storage and
reduction schedule. It may be worthwhile, but it is not a safe lifetime-only
edit. We do not have dispatch-level timings or occupancy/register counters here
to establish its benefit or bottleneck. It is explicitly deferred, not reported
as a failed experiment or as already optimized away.

Likewise, candidate transforms overlap and use different support/bases. A larger
DCT is not the same value as a concatenation of smaller candidate DCTs. Sharing
that arithmetic would require a valid factorization and proof of the strict
floating-point decision contract. Fewer operations with a different accumulation
order would not satisfy this branch's byte-preserving requirements. AC search
remains an important future kernel-optimization target; this disposition does
not minimize its measured cost.

The existing quant-norm sharing is different: the producer and consumers use
the same scalar result and the cost slot is free at that phase. The source
explicitly selects `kQuantNormFromForwardPass` for larger fused candidates, and
the final cost overwrites it only after its consumers finish. It is a qualified
reuse boundary already in the integrated implementation.

## Fresh phase evidence

Artifacts are in `build/resident-reuse-audit/`. `run.py` verifies the sealed
`be00797` checkpoint before collecting three independent host-profile and GPU
stage-profile processes per input. Host/stage order alternates. Each process
uses two warmups and five samples. Inputs are pinned canonical Kodak17 and
padded-stress 3839x2159 PFM, with hashes recorded; the initial built-in gradient
probe is retained separately and is not mixed into these medians.

Configuration: the qualified Release `build/public-admission` binary, M4 Pro,
48 GiB, SIMD/fused-tuned, fully resident, effort 7, target 1.2, CPU automatic,
default accounted domain. The optional compile-time Metal profiling flag remains
off; the stage timestamp API is available. Dispatch timestamps are unavailable.
GPU-profile samples compare their bytes with an unprofiled encoding oracle.
This is an attribution audit, not a new parent/candidate performance experiment.

| Root host-profile boundary, ms | Kodak17 | Stress 4K |
| --- | ---: | ---: |
| Workflow profile total | 23.232 | 252.353 |
| Input preparation | 0.747 | 4.478 |
| Quantization pipeline | 15.620 | 205.225 |
| Codestream encoding | 6.940 | 42.972 |

The root profile excludes the final outer publication epilogue; complete public
call measurements remain in the [admission record](resident-public-admission.md).
Do not add medians to reconstruct a particular call. Serializer `_work` fields
are retained in raw evidence but are not treated as wall-time multipliers.

The separately instrumented GPU-stage runs give these attribution medians:

| Boundary, ms | Kodak17 | Stress 4K |
| --- | ---: | ---: |
| AC prepare, host wall | 0.311 | 3.383 |
| AC readback, host wall | 0.028 | 0.474 |
| AC placement/merge, host wall | 0.225 | 3.250 |
| AQ reconfiguration, host wall | 0.116 | 0.984 |
| Independent frame snapshot/assembly, host wall | 0.034 | 0.752 |
| Reference preparation, command-buffer GPU duration | 0.844 | 18.390 |
| AC search, command-buffer GPU duration | 1.866 | 41.617 |
| Resident AQ, command-buffer GPU duration | 9.296 | 100.372 |

These instrumented spans are not interchangeable with the uninstrumented root
times, and parent wall stages overlap their children. The 4K AC GPU cost is
material, but removing the whole CPU setup would save only that setup span;
caching its matrix subset cannot eliminate AC GPU work. Similarly, making a
small frame snapshot zero-copy is not a reason to retain much larger arenas.

Across every stage-profile sample, all seven selected transform families have
one `aq.reconstruction.forward.*` invocation, two coefficient reconstruction
invocations for effort 7, and a separate final-frame coefficient pass. There is
one reference-preparation submission and one resident-AQ submission. This agrees
with `preserve_forward_coefficients` and the source-recorded policy loop; it
does not prove individual kernel occupancy or bandwidth behavior.

## Retained benefits, costs and closure

The [joint preparation qualification](resident-execution-integration.md) proved
exact parent/candidate outputs and measured the combined shared-storage/deferred
preparation improvement against the handoff parent. The
[last-use qualification](resident-last-use.md) separately measured about 5.1%
lower padded-4K peak physical footprint, with a recorded small-image latency
regression. Those are distinct historical comparisons, not additive speedups or
fresh timing claims for this audit. The implementation remains present and the
current runtime is covered by the sealed public-admission parity/decoder suite.

The [public admission pressure study](resident-public-admission.md) exposes the
retained cache/slot tradeoff: tight 4K single-image admission was about 408 ms
versus 353 ms unlimited in its CPU-one cohort; four-image minimum-fit batches
were about 1,630 ms versus 896 ms with four unlimited slots. The hard boundary
is retained. A transactional cache-credit optimization is a specific future
candidate with proof obligations, not a reason to weaken accounting or promise
that a memory cap is free.

Fresh permanent AQ evaluation, quantization-pipeline, completed-frame,
whole-resident-plan and public-admission tests pass three times each (15 runs).
They retain padded/boundary, diagnostic, alias-lifetime, failure and output
independence coverage. The unmodified runtime's full suite, exact corpus/policy
comparisons and sanitizer limitations are recorded in the verified admission
checkpoint; no tolerance is widened and no new alias is being qualified here.
A fresh pinned-decoder conformance run also passes all 22 fixtures. The final
audit manifest seals 420 source, profile and validation hashes.

Every member of the audited set now has an implemented, intentionally retained,
or explicit deferred disposition with its reason and relevant cost/lifetime
evidence. Milestone 5 is complete. This does **not** complete resident execution:
milestone 6 must coordinate actual CPU participation across jobs, relinquish
capacity at blocking boundaries, preserve fairness and lifetime rules, and
qualify queue/service latency, throughput and memory under concurrency.
