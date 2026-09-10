# Shared AQ/Butteraugli storage and resident host preparation

This change prepares the successful shared-storage and host-cleanup experiments
for normal use. It is on `perf/metal-preparation`, based on `0fc9345`. Local
`main` at `4ea12ab` has the identical committed source tree (the intervening
commit merges the fusion branch), so there is no source-porting dependency.

The candidate enables these storage changes through the existing resident
workflow. It contains no experiment environment switches, retained Butteraugli
arena, asynchronous reference preparation, AQ-policy change, or new public
encoder option. The original experiments remain separately preserved.

## Storage lifetime

The prepared Butteraugli operation can borrow nine F32 planes from its owning
AQ operation. A typed internal descriptor fixes the plane count. Preparation
checks the backing backend, available byte range, disjointness, and absence of
reference overlap before repacking padded storage to the reference dimensions.

| Borrowed storage | Last AQ use before a comparison | Butteraugli use |
| --- | --- | --- |
| Two three-channel filter scratch images | Filtering and conversion into separate reconstructed linear RGB | Psycho-image construction, then AC/DC accumulation |
| Three planes from gathered transform pixels | Forward transform; retained coefficients have separate storage | Convolution and difference intermediates |

Reference preparation completes before AQ starts. Each comparison follows its
reconstruction/filter/color-conversion work in command order. Subsequent AQ
iterations regenerate the borrowed intermediates before reading them. Cached
references, forward coefficients, reconstructed linear RGB and final frame
coefficients remain distinct. Single-scale/expanded images and preparations
without two filter scratch images use owned Butteraugli scratch.

Destruction waits for an outstanding AQ submission, destroys the Butteraugli
borrower, then returns the AQ arenas to their existing lease pool. The existing
failure path discards poisoned AQ leases. There is no additional idle arena.
The multiscale final staging map is allocated at its actual half-width and
half-height; standalone full-map comparisons retain their normal API.

At padded 4K, sharing removes 298,382,976 Butteraugli-owned bytes and packing
removes another 24,859,264 bytes: 308.3 MiB in total. Memory accounting excludes
the borrowed bytes from Butteraugli-owned scratch so the enclosing AQ totals
count each physical allocation once.

## Host preparation and state contracts

The resident workflow initially leaves its host pixel-mask vector empty. The
prepared evaluator also defers its host mask staging. Initial quantization
still returns the small quantization and strategy fields; the AC tournament
consumes the mask from device memory. Its CPU merge accepts an entirely empty
host mask because candidate scoring has already consumed the device mask.
Ordinary CPU search and nonresident GPU search continue to require a mask.

A device scan validates the final blurred mask for positive finite values
before the existing error readback. The raw mask was already validated during
generation, but that does not replace checking the final blurred result.
Malformed partial descriptors are rejected. An explicit mask output is
materialized lazily and committed only after all validation succeeds, including
strided outputs and subsequent calls on the same prepared evaluator.

The resident frontend explicitly requests deferred final transform metadata.
Initial quantization and reference preparation are available in this state.
Evaluation, the resident policy and final CfL preparation reject it until a
successful `Reconfigure` supplies authoritative strategies and metadata. The
ordinary prepared-evaluation API remains eager unless this capability is
requested. Device metadata capacity is planned directly from tile geometry,
without constructing provisional host transform records or tile offsets.

## Qualification

The evidence is retained under `build/preparation-qualification`. Its build
manifest records commands and hashes for identical drivers linked against
fresh Release baseline/candidate libraries. The baseline is `0fc9345`; flags
are `-O3 -DNDEBUG`, tests and benchmarks enabled, libjxl reference oracle and
compile-time Metal profiling disabled. Hardware is Apple M4 Pro, 48 GiB,
macOS 15.6. GPU implementation is SIMD/fused-tuned.

Timing measures the complete encode call including internal teardown, excluding
backend construction, image loading, hashing and output writes. Each process
alternates two changed images. Main matrices use one cold encode, two further
warmups and six retained encodes. Three independent process pairs alternate
baseline/candidate order. Results are per-image process medians and paired
latency ratios; images sharing a process are not independent process samples.

The main corpus contains all 24 Kodak images and six photographs at each of
1080p and 4K. Additional runs cover 16 synthetic extents from 1x1 to padded 4K,
efforts 4/7/8, distances 1.0/1.2, exact coefficients, fresh backends, growing
contexts and memory. A separate small-image follow-up uses longer warmups and
seven process pairs to investigate variability around the size boundaries.

All matrices are complete: 242 primary process runs plus 140 longer small-image
runs. The following medians summarize the natural-image cohort; speed changes
are median paired ratios, not ratios of the two displayed aggregate medians.

| Corpus | Images | Baseline ms | Candidate ms | Median paired latency change | Faster image/process pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak | 24 | 25.89 | 24.94 | -4.04% | 63/72 |
| Photographs, 1080p | 6 | 89.97 | 83.86 | -7.11% | 18/18 |
| Photographs, 4K | 6 | 299.51 | 280.10 | -6.33% | 18/18 |

The displayed times are medians across each image's three process medians.
At 1080p, additional synthetic checks show paired reductions of 8.20% at effort
4, 5.14% at effort 8, and 5.47% at distance 1.0. Fresh-backend and alternating
size checks also pass output parity. These are separate small diagnostic
cohorts, not additional natural-image evidence or a CPU-encoder comparison.

Three memory probes at padded 4K measure median peak physical footprint of
3337.9 MiB for baseline and 2970.7 MiB for the candidate, a 367.2 MiB reduction.
After one second idle with the backend alive, both report about 1192 MiB;
the candidate adds no Butteraugli retention. This idle boundary differs from
the five-second probes in the earlier experiment and is not a free-memory or
resident-set-size measurement.

## Small-image interpretation

The follow-up discards eleven encodes per process and retains eleven encodes
of each changed image, using seven independent process pairs per extent.

| Source extent | Median paired latency change | Faster image/process pairs |
| --- | ---: | ---: |
| 8 × 8 | +1.78% | 5/14 |
| 14 × 16 | +2.63% | 5/14 |
| 15 × 15 | +2.51% | 6/14 |
| 16 × 16 | -3.02% | 9/14 |
| 17 × 19 | -2.48% | 7/14 |
| 63 × 65 | -3.10% | 8/14 |
| 127 × 129 | -4.78% | 12/14 |
| 255 × 257 | -3.29% | 10/14 |
| 511 × 513 | -3.05% | 14/14 |
| 767 × 511 | -2.68% | 14/14 |

Tiny-image process medians vary substantially in both builds, often by several
milliseconds. The first three sizes have slightly slower median paired results;
this is not a universal latency win. The measurements do not establish a stable
size threshold, so the candidate uses structural eligibility (multiscale plus
two available filter images) for sharing and introduces no empirical size gate.
The evidence for larger natural images and reduced allocation is much stronger.

## Correctness and integration status

All 6,742 saved codestreams match their corresponding baseline SHA-256 across
both matrices. Pinned `djxl v0.13.0 e8ff0976` independently decodes 72 outputs:
baseline and candidate for every one of the 36 natural inputs. Each pair's
linear-RGB PFM SHA-256 matches. Input, binary, codestream and decoder hashes,
commands, sample rows and logs are retained in the evidence manifests.

Both fresh baseline and final candidate pass 61/62 full Release tests. The sole
failure is the same existing CPU `quantization_pipeline` golden mismatch:
actual `0.24919039011001587`, expected `0.24914586544036865`. No tolerance or
expected allocation count was changed. The final full suite includes the new
checks for deferred metadata, evaluation/final-CfL rejection before successful
reconfiguration, lazy strided mask materialization, omitted-mask numeric and
readback failures, and malformed-mask rejection before GPU work. Resident AC
merge results match with and without the unused host mask. Production AQ tests
now exercise all eight Gaborish/EPF combinations against the existing CPU
reconstruction oracle, alongside existing concurrency, destruction, poisoned
lease, diagnostic, small/expanded, and maximum-error checks.

The final rebuilt libraries reproduce the qualified candidate driver byte for
byte when relinked with the same executable name. The final diff passes whitespace
checks and a read-only apply check against current main. `integration.patch`,
`completion-audit.json`, and the compact `preparation-evidence.tar.gz` are under
`build/preparation-qualification`. The archive contains the patch, drivers,
commands, manifests, sample/trace logs and report; large JXL/PFM files and binary
artifacts remain in the evidence directory.

The change is qualified for integration on `perf/metal-preparation`. Qualification
left main, the fusion worktree, and the earlier experiment unchanged.
