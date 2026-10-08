# Phase 1 extraction qualification

The extraction implements P1.0–P1.4 of the
[architecture plan](../modular-architecture.md). VarDCT remains the only complete
image encoding mode. The shared subset supports global trees, default weighted
prediction parameters, no transforms, and the existing prepared entropy models.

## Implemented boundary

| Location | Responsibility |
| --- | --- |
| `codec/modular/prediction.h` | Existing default weighted arithmetic/state, with the original preparation/serializer owner parameter |
| `codec/vardct/dc_prediction_internal.h` | Explicit mapping from the unchanged public `VarDctDcPrediction` enum |
| `codec/vardct/dc_quantization.cpp` | Lossy DC decisions and reconstructed DC, using the shared predictor |
| `codestream/modular/stream_types.h`, `stream_encoder.*` | Validated four-bit stream suffix, global model emission and borrowed-token emission |
| `codestream/modular/tree_codec.*` | Resolved tree-token emission and checked tree storage planning |
| `codestream/vardct/dc_group.cpp`, `weighted_dc.cpp` | Specialized channel scans, Y/X/B order, metadata packing, context policy and adapter prefixes |
| `codestream/vardct/dc_context_tree*` | Fixed/adaptive DC layout and legacy tree data |
| `codestream/vardct/headers.cpp`, `encoder.cpp` | VarDCT syntax, group-dependent tree split, model preparation, policy and assembly |
| `codestream/headers.cpp`, `fields_internal.h` | Shared primitive field/size writing |
| `codestream/entropy.*`, `entropy_internal.h` | Shared entropy machinery, including the existing validated-model fast emitter |

The generic directories import no VarDCT frame, quantizer, context policy, GPU
backend, or libjxl headers. Gradient loops remain specialized because the codec
and serializer have different integer-width contracts. There is no generic
image copy, per-pixel dispatch, per-group model rebuild, or new allocation owner.
The original predictor arithmetic is unchanged.

Quantization retains `ResourceClass::kCount`, inheriting its caller's resource
context (preparation in the workflow). Tokenization keeps the explicit
serializer class. Tests cover both fixed classes and inherited preparation,
including failure at each of the predictor's five allocations.

The stream suffix is validated before writing. Local trees, custom weighted
parameters, transforms, and unknown enum values are explicitly rejected.
DC precision and metadata anchor counts stay in VarDCT's atomic allotments.
Global tree/model helpers are internal transaction composition functions: their
caller owns the temporary writer/allotment and exception boundary. The existing
`WriteDcGlobalWithLayout` transaction preserves the public atomicity contract.

All existing public headers remain at their original paths with unchanged
types, symbols and defaults. New headers are private. Exported components stay
`gjxl::core`, `gjxl::codec`, and `gjxl::codestream`. The Windows direct-oracle
linkage adjustment is confined to optional test targets. Historical experiment
manifests retain their original source paths; current documentation links and
the source-including validation test use the new paths.

## Baseline and exact comparisons

The detached baseline is `9ca5da9`, whose runtime sources match `67caa6d`.
It was built and captured before editing production sources. Baseline and
candidate use the same Release x64 MSVC 19.37.32826/STL 202305 configuration,
with C++20, Metal/CUDA disabled, and the unmodified storage compatibility guard.
The host is Windows on Snapdragon X2 Elite (12 logical processors); the x64
binaries run under Windows emulation. These are not native ARM64 measurements.

The [protocol](PROTOCOL.md) records owners and timing criteria fixed before
extraction. Reproduction commands are in the
[capture tool README](../../tools/modular_extraction/README.md).

- [Original capture](baseline-artifacts.json): 108 frozen artifacts.
- [Original comparison](comparison.json): all 108 match exactly.
- [Extended comparison](stage-comparison.json): all 204 match exactly, adding 96
  direct numerical DC captures against the untouched baseline libraries.
- [Photographic stage comparison](photo-comparison.json): four additional
  artifacts match for gradient/weighted encoding of one prepared CPU frame.
  The combined correctness set is 208 artifacts.
- [Final native-runtime comparison](final-binary-comparison.json): the original
  108 expectations also match the final native libraries used for timing.

The original matrix contains 40 completed-frame cases and 14 full CPU workflows:
gradient/weighted prediction; adaptive/legacy 1/2/4/34-leaf tree choices;
single/multiple DC groups; mixed strategies; all entropy policies; sampled/full
orders; integer-mapping search; int8/int16 compact and sparse frames; efforts
1/4/7/8/9/10; target-size retries; and a 2500x500 tiled photographic image.
It compares bytes, DC/metadata tokens, tree/context data, entropy decisions, and
the exposed serializer/tokenization storage bounds. Those bounds are identical.

The additional DC captures cover ordinary and prediction-aware quantization,
both predictors, precision 0–3, padded rows, and extents 1x1, 1x257, 257x1,
17x19, 255x256, and 259x259. Quantized coefficients and reconstructed float bit
patterns match exactly, including row padding. These supplemental captures did
not replace or alter the original baseline expectations.

## Tests and retained failures

- Baseline CTest: 113/121 passed; eight pre-existing failures/not-run tests.
- Candidate CTest: 115/124 passed; the same eight, plus the optional direct DC
  processing oracle failure reproduced independently against the baseline.
- New `modular_stream` tests pass: aligned/unaligned header syntax, all precision
  and anchor-count boundaries, unsupported values, atomic failure/recovery,
  predictor allocation ownership, partial-construction cleanup, and composed
  tree bounds under allocation failures.
- Existing weighted-predictor comparison with pinned libjxl passes, as does the
  direct tree oracle. Both baseline and candidate pass all 23 pinned-decoder
  conformance fixtures, including complete CLI workflows.
- Existing resource/admission, failure-recovery, worker-launch/joining,
  concurrency, compact/sparse, entropy, serializer and workflow tests retain
  their baseline passes. Installed C11 and C++20/C++23 consumers pass, including
  individual installed-header compilation and package relocation checks.

The pre-existing Windows failures are `earlier_dc_contract` and
`entropy_readiness_contract` (POSIX environment calls; the former also has an
MSVC initializer diagnostic), five Butteraugli reference/differential targets,
and the generated Butteraugli goldens test (reference link/build failures).
These were captured before extraction; they are not silently excluded from the
reported CTest totals.

The optional `dc_processing_oracle` uses the actual pinned libjxl algorithms
built with x64 clang-cl. It fails at 255x256, channel 2, x=8, y=2, pattern 3:
reference -5850 versus native -5852. Linking the same unchanged test object
against the frozen baseline codec reproduces the identical failure. Its cause
has not been diagnosed here; neither extraction parity nor decoder conformance
is a claim that this independent quantization oracle passes on this toolchain.
The pinned revision is `e8ff09762481785938d8e4e01333ed3917571161`.

## Performance and qualification limits

The [initial seven-pair run](timings-initial.json) covers 94 case/stage
combinations after a warmup pair. It flagged 19, with substantial variability in
some small serializers. The [fresh seven-pair confirmation](timings-confirmation.json)
cleared 18; one 16x16-block gradient/rate-optimized serializer still exceeded the
threshold. A [further seven-pair run with 31 repetitions per pair member](timings-repeated.json)
retained 217 samples per binary for that case: baseline median 3.4054 ms,
candidate 3.4821 ms (+2.25%, +0.0767 ms). It therefore does not exceed both 5%
and 0.05 ms. Its raw median absolute deviations were 0.2228/0.2027 ms.

No material regression remains under the predeclared criteria. This is a
same-host extraction gate, not a speedup claim. Earlier flags and every sample
remain in the records; unrelated wins were not averaged against them.
No timing from concurrent build/test runs is used for the gate.

| Representative ordinary call | Baseline median | Candidate median | Change | Record |
| --- | ---: | ---: | ---: | --- |
| 260x260-block weighted DC tokenization | 12.843 ms | 13.269 ms | +3.32% | Initial |
| Same frame, complete serialization | 38.486 ms | 38.326 ms | -0.42% | Initial |
| 65x49 CPU workflow, effort 1 gradient | 44.241 ms | 44.580 ms | +0.77% | Confirmation |
| 65x49 CPU workflow, effort 7 weighted | 81.712 ms | 82.598 ms | +1.08% | Confirmation |
| 2500x500 photographic CPU workflow | 12.847 s | 12.961 s | +0.89% | Initial |

The [photographic stage run](timings-photo.json) adds seven pairs with three
repetitions each, using one prepared fixed-DCT8 CPU frame per process. Both
predictors pass the stage gates: gradient DC 1.172→1.170 ms (-0.11%) and
serialization 25.925→24.870 ms (-4.07%); weighted DC 3.497→3.607 ms (+3.15%) and
serialization 28.428→28.002 ms (-1.50%). Together the initial and photographic
runs cover 98 case/stage combinations. Preparation is outside these stage
timers; the complete workflow above includes its preparation cost.

[The manifest](manifest.json) records runtime source hashes, compiler/package
identities, fixture and binary hashes, full test outcomes, the baseline oracle
link command, dependency closure, and hashes of retained build/test logs.

Metal/CUDA-enabled execution, native ARM64, other operating systems, and
sanitizer configurations were not run on this host and remain unqualified.
No GPU kernels or backend dispatch changed. Phase 2's general image channels,
learned/local trees, configurable prediction, transforms and whole-image Modular
workflow remain future work.
