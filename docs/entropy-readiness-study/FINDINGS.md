# Findings: earlier population reduction and entropy readiness

**The mechanism works and is worth developing for single-image ordinary encoding with spare CPU capacity. The current static split should not become a blanket default.** The experiment preserves exact outputs, but benefits depend on which entropy branch is critical and whether reserving a participant reduces DC throughput.

[Full report](REPORT.md) · [Implementation and remaining work](DESIGN.md) · [Clean review patch](prototype-clean.patch) · [Protocol](PROTOCOL.md)

## Measured complete-call gains

Positive means lower latency. Each result below uses four processes and 72 counterbalanced measured blocks. Battery and AC results are separate; no adjustment or stage subtraction is applied to public-call latency.

| Input / effort | Battery: earlier Finish + entropy | AC confirmation: earlier Finish + entropy |
| --- | --- | --- |
| CLIC A 3.09 MP / e1 | +2.224 ms / +10.70% | +1.711 ms / +10.86% |
| CLIC A 3.09 MP / e7 | +2.361 ms / +3.51% | +1.817 ms / +3.23% |
| CLIC B 3.36 MP / e1 | +2.440 ms / +8.96% | Not captured |
| Campus 12 MP / e7 | -6.796 ms / -2.52% | +1.495 ms / +0.81% |
| Alpine 24 MP / e1 | +1.793 ms / +1.88% | +0.564 ms / +0.76% |
| Alpine 24 MP / e7 | -15.108 ms / -2.84% | +1.575 ms / +0.50% |

The battery matrix recorded regressions at large effort 7. In the separate battery follow-up for Campus, the median paired quantization-phase difference was a 16.988 ms slowdown while serialization improved by 3.020 ms. Quantization precedes the intervention. That prevents assigning the whole-call regression directly to the new entropy schedule; background activity, power behavior and cross-call effects are not isolated here. The raw regressions remain in the results. AC confirmation is an independent condition, not a replacement for them.

## Why earlier entropy adds more than earlier Finish alone

On CLIC A/effort 1, AC-powered earlier Finish alone saves **+0.999 ms / +6.19%**; releasing entropy tasks separately saves **+1.711 ms / +10.86%**.

Moving Finish allows population reduction to run while DC token preparation is still active. But an all-tokens barrier can still keep the AC model waiting after reduction. Mode 2 removes that second dependency: DC modeling follows DC preparation, and AC modeling follows Finish, while the order model uses its already-ready tokens. Selection and writing still join at their original boundary.

| AC trace case | Variant | GPU-ready → reduction start | GPU-ready → AC model start | DC-ready → DC model start |
| --- | --- | --- | --- | --- |
| CLIC A / e1 | Control | 1.868 ms | 2.870 ms | 1.114 ms |
| CLIC A / e1 | Earlier Finish | 0.069 ms | 1.924 ms | 0.021 ms |
| CLIC A / e1 | Earlier entropy | 0.075 ms | 1.167 ms | 0.000 ms |
| 24 MP / e7 | Control | 1.053 ms | 2.293 ms | 1.203 ms |
| 24 MP / e7 | Earlier Finish | 0.116 ms | 1.400 ms | 0.279 ms |
| 24 MP / e7 | Earlier entropy | 0.098 ms | 1.291 ms | 0.000 ms |

Population reduction is still CPU work. It was moved, not eliminated or ported to the GPU. The two branches call the same model-building helper and preserve coefficient-order/context decisions. GPU shader bytes are unchanged. The [timelines](REPORT.md#what-changed-in-the-measured-dependency-chain) show actual overlap.

## CPU capacity changes the answer

With only two CPU participants, the 24 MP/effort-1 case is **8.107 ms / 7.64% slower**. One participant is reserved for AC completion/modeling, leaving DC with one. The corresponding diagnostic traces show:

| CPU=2, 24 MP/effort 1 | Control | Earlier entropy |
| --- | --- | --- |
| Distinct DC group threads | 2.000 | 1.000 |
| DC preparation | 9.530 ms | 18.720 ms |
| Population reduction | 1.123 ms | 1.141 ms |

CPU=1 and unavailable-admission cases fall back to the old schedule. Their nonzero measured differences are noise controls, not optimization gains. A waiting AC worker retains protected capacity; resetting the local nesting depth preserves DC group parallelism but cannot restore the participant reserved for AC. A production design should consider DC group capacity before splitting, or schedule ready continuations on a shared pool.

## Batches and high efforts need separate policy

| Repeated-image B4, CPU limit 8, AC power | Earlier Finish + entropy: batch makespan reduction |
| --- | --- |
| CLIC A / e1 | -0.131 ms / -0.38% |
| CLIC A / e7 | +4.086 ms / +2.03% |
| 24 MP / e1 | -3.260 ms / -1.05% |
| 24 MP / e7 | +4.400 ms / +0.30% |

The batch driver already fills some idle windows with other images. Reserving a participant can compete with those images, so single-image savings do not imply batch throughput gains. These batches repeat one image; heterogeneous batches are outside the experiment.

Efforts 8–10 were tested with completion overlap only. The full early-entropy variant deliberately retains the existing dual-representation schedule there. Its apparent differences in those rows are another unchanged-path noise control. Moving both representations earlier needs a separate bounded task graph that preserves complete-file selection and balanced fallback.

## Recommendation

Continue #1 as a targeted scheduling change. Start with ordinary single-image paths where AC and DC/model work have usable overlap and the extra participant does not materially reduce DC throughput. Preserve the old schedule when admission is unavailable. Do not select a production resolution threshold from these few images.

Before enabling it by default, integrate native profiling of the new overlaps, settle CPU/batch admission policy, document the additional live dispatcher storage, and qualify a build without measurement hooks. The high-effort dual representation case remains an explicit follow-up. [Design details and review patch](DESIGN.md) are retained.

Validation: **160/160 Release tests**, **84/84 scheduling contract cases**, **200 capture processes**, **10106 encoded image instances**, and **35 independently decoded references**. All compared bytes and summaries match; CPU caps and post-call cleanup checks pass. Warm non-forced token retries: **0**. Forced retries are separately retained. At study closure, the original checkout and index were unchanged. This evidence package records that completed study; it does not enable the prototype.
