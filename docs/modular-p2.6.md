# P2.6: frozen CPU oracle and GPU handoff

P2.6 is implemented on `feat/modular`, following P2.5 commit `dda606d`.
The [architecture](modular-architecture.md) now links the retained
[fixture protocol](../tools/modular_oracle/README.md) and concrete
[GPU handoff](modular-gpu-handoff.md). No GPU kernels or dispatch changes are
part of this milestone.

The scalar path remains available with explicit transforms, predictors, tree,
weighted parameters and entropy mode. A versioned manifest freezes 286 stage
and file artifacts across seven corpus categories and both coders. The corpus
covers all six integer profiles, transparent color, clipped multi-group scans,
prescribed RCT/palette/squeeze and weighted split-tree execution. It records
unpacked and transformed planes, resolved policy, tree tokens, every prediction,
property and leaf decision, residual/context tokens, HybridUint populations,
normalized models, payload bits/bytes and complete files. Learner decisions are
separate policy artifacts. Broader boundary and extrema coverage remains in the
existing conformance, policy and transform suites.

The independently linked capture checks transformed values, shapes/shifts,
tree tokens, all 14 predictors and 16 properties, residual/context tokens and
complete decoded samples against pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161`. Exact palette ordering uses an
independent ordered dictionary with pinned metadata application. The native
capture runs without libjxl. Both match the same checked-in SHA-256 manifest.
Every prescribed file also matches with profiling enabled and four participants.
Tests never rewrite expected hashes.

Private, opt-in caller timers cover input preparation, transforms, training,
tokenization, data-model optimization, emission and assembly. Search records
the chosen policy and candidate count; replaying the chosen policy must produce
the same complete file. Profiling adds no managed allocations, workers, public
API fields or installed headers. It preserves existing arithmetic and policy.

## Qualification

The [machine-readable record](modular-p2.6-results.json) contains source/binary
and evidence hashes, commands, test outcomes and retained limitations. Local
logs and regenerated artifacts live under `build/modular-p2.6/`.

- Broad CTest: **125/134 passed**, including all 11 Modular tests, managed
  allocation/worker failure recovery, C publication, mixed-domain concurrency,
  CLI, installed C/C++ consumers and package relocation. No P2.5 pass was lost.
- Frozen byte regression: **2,802 default**, **1,646 policy**, **626 transform**
  Modular files and **208 VarDCT** artifacts unchanged.
- CPU-only Release build/install with reference, Metal and CUDA disabled passed.
  The installed native capture reproduces all **286 frozen artifacts** using
  matching private source headers and installed libraries, without exporting
  private headers or linking reference libraries.
- Rust workspace: **12 tests passed**. Pinned independent VarDCT decoding:
  **23/23 passed**.

The same nine baseline failures/unavailable tests remain: `earlier_dc_contract`,
`entropy_readiness_contract`, `dc_processing_oracle`, `butteraugli_reference`,
the four Butteraugli blur/frequency/distance/scalar-distance differential tests,
and `butteraugli_goldens_generated`. The broad build retains the same eight
pre-existing failed targets. The DC oracle still reports the known 255x256,
channel 2, (8,2), pattern 3 discrepancy (reference -5850, native -5852).
These failures are not hidden or reported as passing.

Qualification used Release x64 MSVC 19.37.32826/STL 202305 on Windows ARM64
under x64 emulation, SDK 10.0.26100.0; the pinned reference uses x64 clang-cl.
Native ARM64, native x86-64 hardware, other operating systems, sanitizers and
GPU-enabled builds remain unqualified here. Phase 2's supported CPU profile is
complete with these explicit platform and inherited regression limits.

## Performance baseline

[Repeated measurements](modular-p2.6-timings.json) retain first-call latency,
three warm samples per native configuration, median/min/max, megapixels/second,
encoded size, observed managed backing peak, conservative planned capacity and
per-stage medians. There are 56 native configurations: seven categories,
prefix/ANS, prescribed/search, and one/four participants. Selected policies and
files are retained and hashed; serial/parallel policies and complete bytes agree.
Reference effort 1 is measured on the same inputs with lossless and
keep-invisible enabled, without a parallel runner. Reference memory is unmeasured.

The final measurement run follows the completed regression/build jobs. Run 0
is the first call for that configuration in an already running process, not a
claim of cold caches. Fixture generation, independent checks, artifact writes
and domain construction are outside complete-call timing. Admission, encoding
and output publication are inside it. Named stages are exclusive caller wall
intervals, including joined workers; search includes all losing candidates.
Managed capacity excludes caller storage, stacks and capture/reference tooling;
it is not process RSS. Details and commands are in the fixture protocol.

For the prescribed, serial prefix configurations (warm medians):

| Input | Complete call (ms) | Encoded bytes | Managed peak (MiB) | Planned capacity (MiB) |
| --- | ---: | ---: | ---: | ---: |
| tiny | 0.14 | 21 | 0.03 | 0.26 |
| gray16 | 7.34 | 48,350 | 3.17 | 30.24 |
| screen | 35.17 | 152,680 | 9.49 | 89.61 |
| palette | 8.71 | 20,283 | 3.53 | 30.27 |
| noise | 40.13 | 828,702 | 9.49 | 89.59 |
| alpha16 | 53.43 | 284,790 | 12.66 | 119.28 |
| photo | 20.35 | 129,197 | 4.19 | 53.34 |

These measurements establish a baseline, not a speed or compression parity
claim. Native search settings do not correspond to libjxl effort numbers.
Weighted tokenization, model optimization and repeated search work contribute
differently across the corpus. The handoff therefore begins with a small exact
integer unpack/RCT contract for qualification, then evaluates nonweighted
tokenization and fixed-model work, with a separate strategy for weighted state.
Keep learning, global model decisions and assembly on CPU until measurements
justify moving them. Transfer-inclusive measurements on actual target hardware
must precede automatic GPU routing.
