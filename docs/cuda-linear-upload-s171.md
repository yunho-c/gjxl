# S171: contiguous original-RGB upload routing

## Scope

Starting revision is `f86c905be7c16fa55a3b63a64504d4ccf8bebb85` (S170).
Production remains S168's runtime: default fully resident CUDA, native sparse
ownership enabled and compact-width packing disabled. S171 investigates the
original RGB upload, not a new coefficient representation, input cache,
pinned allocation/registration scheme, numerical approximation or tile kernel.

Artifacts are under `U:/gjxl-cuda-diagnostics/s171-artifacts`, with ignored
`build-cuda-ninja/profiles/s171_*` helpers and archived production sources.
The complete 1,053-file S170 artifact inventory and 343 production files were
verified before preparing the experiment. No predecessor is modified.

## Mechanism and prototype

S169's ordinary 4K input-preparation phase is roughly 17 ms. Its separate
traces identify three original-RGB uploads totaling 99,460,812 bytes and
15.293–15.807 ms of transfer activity across the two 4K image cases. These
historical observations motivate investigation; they are not candidate timing
evidence or a demonstrated PCIe bandwidth ceiling.

`CudaPreparedLinearRgbOpsin::Encode` always calls `cudaMemcpy2DAsync` for each
source plane. The destination is allocated with a tight row stride. In the
ordinary owning-image workflow the source is tight too, although public
views also support padded and different per-plane strides. Separately, the
existing generic `CudaBackend::CopyHostToDevice2D` already selects
`cudaMemcpyAsync` when both pitches equal the active row width.

The diagnostic includes the exact archived input-owner implementation and
replaces only its local upload call with a private selector:

| Mode | Upload route |
| --- | --- |
| 0 | Existing 2D copy |
| 1 | Identical control, the same 2D-copy instruction path |
| 2 | Contiguous copy if both pitches equal row bytes; otherwise existing 2D copy |

The wrapper audits call count, route count, mode and logical uploaded bytes.
It has a common multiplication-overflow check; the production owner has
already validated the allocation geometry. No storage, synchronization,
prepared-owner lifetime, caller-source borrowing or failure handling is
changed. The existing operation stream and submission wait are retained.
Supporting genuine strided views is required functionality, not a new
compatibility layer. Mode selection exists only in ignored diagnostics.

Normal and ASAN fixture executables and whole-encoder DLLs each contain all
eleven byte-identical standard S169/S168 GPU modules. No CUDA kernel is
recompiled or replaced. Normal links current S168 codestream support and
S166's other normal libraries. ASAN reuses the qualified S168 bit writer,
S167 ANS writer, S166 encoder and S162 instrumented support libraries, with
the unchanged uninstrumented PFM reader. The new input owner and harnesses
are compiled normally or with host ASAN as appropriate.

## Guarded preparation qualification

The full fixture covers thirteen geometries:
`1×1`, `1×17`, `17×1`, `7×9`, `8×8`, `31×33`, `65×71`, `257×17`,
`513×519`, `1919×1079`, `3839×2159`, `4097×1`, and `1×4097`.
The quick CUDA-sanitizer subset is `1×1`, `7×9`, `65×71`, `257×17`.
For each geometry, it checks all three modes, three source layouts (tight,
uniformly padded, mixed per-plane pitches), statistics off/on, and two
changed-input generations. Guard regions and row padding contain NaNs;
finite active pixels include negative zero.

Downloaded original pixels must match source bytes, including signed zero.
Padded opsin must match both the retained CPU transform and mode 0 bit-for-bit;
matrix-scale statistics match the CPU oracle and baseline exactly. The source,
row padding and surrounding guards must remain byte-identical. Copy audits
prove mode 2 uses three linear calls for tight input, none for uniformly
padded input, and one for mixed pitches.

Further tests start with an existing prepared owner, reject invalid padded
geometry, zero/infinite intensity, too-short stride, active NaN and active
infinity, then successfully prepare again on the same backend. Failed
preparation must clear the output owner; geometry errors issue no uploads,
and GPU-detected nonfinite inputs issue all three. Null output and invalid
diagnostic mode are also rejected. These are the enumerated failures, not
an allocation-failure injection or concurrent-source-mutation test.

Normal and ASAN full fixtures each pass 504 successful preparation checks
and 37 rejections. Each of four CUDA sanitizers passes 180 successful checks
and 37 rejections: **1,728 successful checks and 222 expected rejections**
overall. Memcheck/initcheck/synccheck report zero errors, racecheck reports
zero hazards, and memcheck reports zero leaked bytes.

The original host build stopped before any fixture or encode: MSVC reported
`C2375` for inconsistent diagnostic export declarations. Original sources,
inputs, log and rejected job remain intact. Separate v2 headers expose the
DLL-loader function-pointer types; direct fixture declarations are local to
the fixture translation unit, while exports are defined in the hook.
Normal and ASAN v2 builds pass without warning or sanitizer suppression.

## Complete-encoder protocol

The eleven established S169 cases are retained: six frozen image fixtures
and five deterministic synthetic cases. Larger Flower/Keong images are
nearest-replicated photographs, not independent native-resolution images.
Settings are fully resident CUDA, effort 7, final score disabled, distance
1.2 except synthetic pattern 2 at 0.01, and separate automatic/eight-thread
CPU budgets. All input and frozen-oracle hashes are pinned.

Every process first encodes with the frozen S168 normal DLL. Image fixtures
also match their frozen codestream files. Every subsequent encode must match
reference bytes, the complete public summary, coefficient width and native
storage size. The hook must report three original-image uploads and exactly
`width × height × 12` bytes; mode 2 must report three contiguous copies.
All 41 host phase durations must be finite and nonnegative.

Qualification covers 44 normal/ASAN case/budget processes, then whole-encoder
CUDA memcheck on 4K and initcheck on Keong/eight threads. Each process has
one reference and one three-mode round: 46 processes / 184 exact encodes.
Both whole CUDA checks pass with zero errors; memcheck has zero leaks.
These preflight durations are not included in performance summaries.

The ordinary campaign uses 44 processes: eleven cases × two budgets × two
repeats. Each has six warmup and eighteen measured three-mode rounds, in six
orders balanced for mode position and directed predecessors. The case/budget
shuffle uses seed `17120260910`; the second repeat reverses that schedule and
order traversal. There are 3,168 diagnostic encodes and 44 references, with
2,376 measured encodes. Mode 0/1 share the exact existing route, providing a
within-process control for timing variation.

Whole-call wall time includes the complete bridge `Encode` workflow and its
internal frame lifetimes. Reference encoding, input-file loading, image/backend
construction, clearing previous/final results, queries, exact comparisons,
logging and NVML calls are outside the timer. Input preparation is the
mechanism-specific host phase; later nested phases and worker-work sums must
not be added as disjoint wall time. Pairwise median ratios need not be
algebraically transitive. No sample filtering, automated retry, setting change,
or historical-versus-current causal comparison is permitted by the protocol.

## Results and decision

All 44 ordinary processes completed from 02:32:48.992452 to
02:39:36.631489 UTC on September 10, 2026. All 3,212 encodes pass, including
2,376 measured encodes, and all 6,336 warmup/measured NVML endpoints report
40,000 mW. No builds, sanitizers, profiler captures or archive compression
overlap the ordinary campaign. No sample or process was discarded or rerun.

The following ranges span four separate process cells per input (two budgets
× two repeats). Changes are median within-round paired differences/ratios,
not ratios of aggregate medians or confidence intervals. Negative means faster.
The preparation table is mechanism-specific; the whole table keeps later
workflow variation visible. Counts of favorable cells are not significance
tests. Full per-process distributions and all phase fields are retained.

| Input | Baseline preparation ms | Candidate preparation delta ms | Candidate preparation change | Control preparation change | Faster than both / 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1025_p2 | 2.833 to 2.947 | -0.069 to +0.016 | -2.30 to +0.54% | -1.28 to +2.13% | 2 |
| 1080p | 4.630 to 4.667 | -0.019 to +0.003 | -0.41 to +0.07% | -0.62 to +0.31% | 2 |
| 4k | 17.372 to 17.627 | -0.335 to -0.044 | -1.89 to -0.25% | -0.99 to +1.33% | 4 |
| 513_p0 | 1.065 to 1.093 | -0.031 to +0.004 | -2.79 to +0.41% | +0.08 to +2.48% | 3 |
| 513_p1 | 1.123 to 1.158 | -0.035 to -0.005 | -3.08 to -0.50% | -2.56 to +0.62% | 3 |
| 513_p2 | 1.129 to 1.158 | -0.025 to -0.010 | -2.26 to -0.92% | -0.47 to +4.98% | 4 |
| 65_p2 | 0.216 to 0.233 | -0.023 to +0.005 | -10.50 to +2.27% | -1.36 to +7.74% | 3 |
| flower_2000 | 8.714 to 8.788 | -0.074 to +0.029 | -0.84 to +0.33% | +0.01 to +2.50% | 2 |
| flower_3200x2160 | 14.526 to 14.611 | +0.081 to +0.226 | +0.56 to +1.56% | -0.40 to +1.29% | 0 |
| flower_500 | 0.863 to 0.909 | -0.003 to +0.022 | -0.34 to +2.61% | +0.32 to +2.76% | 1 |
| keong_3839x2159 | 17.422 to 17.647 | -0.255 to +0.121 | -1.45 to +0.69% | -1.66 to +0.51% | 1 |

| Input | Candidate whole vs baseline | Candidate whole vs duplicate | Identical control vs baseline |
| --- | ---: | ---: | ---: |
| 1025_p2 | -1.92 to +4.70% | -1.92 to +0.22% | -5.39 to +2.47% |
| 1080p | -2.99 to +1.41% | -2.36 to +0.20% | -2.95 to +1.48% |
| 4k | -3.61 to +1.95% | -3.18 to +2.85% | -2.11 to +1.71% |
| 513_p0 | -0.05 to +0.26% | -1.12 to +0.34% | -0.34 to +1.55% |
| 513_p1 | -1.91 to +2.32% | -2.10 to +1.27% | -1.77 to +0.61% |
| 513_p2 | -2.50 to +0.84% | -3.59 to -0.02% | -1.56 to +0.97% |
| 65_p2 | -1.09 to +0.98% | -0.37 to +0.55% | -1.19 to +0.18% |
| flower_2000 | -7.37 to -0.25% | -5.15 to +2.42% | -10.31 to +4.86% |
| flower_3200x2160 | -0.06 to +3.93% | -0.15 to +3.13% | -1.39 to +2.66% |
| flower_500 | +0.12 to +3.43% | +0.26 to +1.77% | -0.77 to +3.92% |
| keong_3839x2159 | -4.63 to +2.62% | -2.18 to +1.14% | -5.05 to +3.26% |

Decision: **do not promote the contiguous-copy branch**. The source-level
simplification does not establish a generally better resident upload policy.
There is favorable preparation evidence for 4K (all four cells beat both
controls, with 0.044–0.335 ms saved), but Flower 3200×2160 is 0.081–0.226 ms
slower in all four cells and the equally sized Keong input is mixed. A
geometry-only promotion rule is not justified by those observations.

The favorable 513/p1 and 513/p2 preparation results are also retained; they
do not establish a whole-workflow improvement. For example, 513/p1 whole
changes range from −1.91% to +2.32%, and 513/p2 from −2.50% to +0.84%.
Flower 2000's −7.37% to −0.25% whole result versus baseline is not reliable
upload evidence: the identical-route control spans −10.31% to +4.86%, the
candidate is mixed versus that control, and preparation changes are small
and mixed. Do not attribute those later-phase differences to the upload API.

No input beats both controls on whole latency in all four process cells.
This does not prove equivalence or a fundamental PCIe limit; it is insufficient
evidence for promotion, including an input-size-specific policy. The generic
row-copy helper's existing contiguous branch is outside the experiment and
is not removed. No production file, public API, compatibility path, staging
allocation or system setting changes.

This checkpoint rules out treating a tight-pitch API substitution as an
established upload speedup on the measured resident workflow. Pinned staging,
registration, caching or upstream fusion would be different mechanisms with
their own setup, lifetime, error-handling and full-workflow costs; S171 does
not qualify or reject them. Higher-cost resident computation and native
coefficient consumption remain in scope. The backend is not established
to be maxed out.

## Preservation and verification

The checkpoint has 98 terminal job journals: 97 accepted and one rejected
initial host build. The 90 encoder jobs account for 3,396 completed exact
encodes, including the preflight/sanitizer cohort. The six preparation-fixture
jobs account separately for 1,728 successful preparations and 222 expected
rejections. The compile failure ran no fixture or encoder. No allocation,
concurrency, decoder-quality, CTest or installation qualification beyond the
explicit tests above is newly claimed for this diagnostic-only change.

`s171_verify.py` checks predecessor and source/link-input pins, binary/native
module hashes, every job log, sanitizer summaries, every parsed sample and
upload-route audit, ordered NVML endpoints, and independently recomputed
whole/preparation paired comparisons and report tables. Final sources,
decision and the complete non-temporary artifact inventory are preserved in
`final_sources.json`, `final_summary.json` and `artifact_hashes.json`.
Use `python -B build-cuda-ninja/profiles/s171_verify.py --frozen` to verify
the archive without requiring future production source identity.

All original and corrected diagnostic versions remain recoverable. No
artifact is deleted or overwritten, no predecessor is modified, and the
three protected user scratch files are not opened, edited or staged.
Only this report is committed; the broader optimization goal remains active.
