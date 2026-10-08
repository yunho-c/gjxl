# Modular extraction qualification protocol

Frozen production baseline: `9ca5da9` (the architecture-plan commit; runtime
sources match `67caa6d`). Capture sources are compiled separately against the
baseline and candidate libraries and matching headers. Never copy candidate
runtime objects or generated expectations into the baseline build.

## Required comparisons

- Hash every complete codestream, DC/metadata token capture and decision/storage
  record. Compare baseline and candidate exactly, excluding timing records.
- Exercise gradient and weighted prediction, adaptive and legacy trees, all four
  DC leaf counts, mixed strategies, single/multiple DC groups, entropy policies,
  sampled/full coefficient ordering and nonzero DC values.
- Include compact and sparse completed frames and complete CPU workflows across
  low/high efforts, DC quantization/smoothing choices and target-size control.
- Run existing pinned-reference, conformance, storage/failure, concurrency and
  installed-consumer tests. Record baseline failures independently.
- Record source/compiler/library/tool revisions, CMake configuration, commands,
  case counts, hashes and unavailable hardware configurations.

## Timing criteria fixed before extraction

Correctness captures may run while builds are active; qualification timings may
not. Use the same machine, Release configuration and input/settings for both
binaries. Warm each binary, then run at least seven paired baseline/candidate
rounds, alternating execution order. Retain individual samples and compare
per-case medians for standalone DC tokenization, serialization and complete
ordinary workflows. Profiled stage times are secondary evidence because detailed
profiling uses a different entropy schedule from the ordinary workflow.

A median regression is material if it exceeds both 5% and 0.05 ms for standalone
DC/serialization, or both 3% and 0.2 ms for a complete encode. A flagged result
requires a fresh paired confirmation under an idle host; it cannot be waived by
averaging it with unrelated wins. If noise prevents a conclusion, retain that
limitation and continue measurements rather than claiming performance parity.
Record small and photographic multi-group cases separately. These thresholds
are extraction gates, not performance claims or future Modular tuning targets.

## Owners crossing the extracted boundary

| Owner | Existing lifetime and required preservation |
| --- | --- |
| Completed frame/channel backing | Borrowed through all synchronous serializer workers; no new channel materialization |
| GPU host handoff | `gpu/ops/quantization_pipeline.cpp` retains the completed frame; `codestream/workflow.cpp` borrows its view into `EncodeVarDctCodestreamToBuffer`. The frame owner survives serialization and worker joining; backend execution remains hardware-unqualified on this host |
| Prediction-aware quantization candidate and predictor rows | Inherit the caller's resource class (`kCount`), normally preparation in the workflow; candidate remains private until successful quantization |
| DC tokenizer predictor rows | Serializer owner; reused between channels and bounded per active group worker |
| DC/metadata token arrays | Existing group storage, unchanged order and capacities; shared model inputs borrow these arrays |
| Resolved fixed-tree tokens | Stack copy with VarDCT group split already patched; generic tree emitter borrows it |
| Tree optimizer and entropy model | Existing serializer allocation/search lifetimes; no additional model per group |
| Global-section temporary writer | Existing caller-owned atomic temporary; extracted helpers append within that transaction |
| Stream header writes | Existing bounded writer allotments, including VarDCT precision/anchor prefixes |
| Final section/output backing | Existing serializer/publication ownership, retry overlap and batch retention |

Public header paths, function/type identities, export targets, allocation-failure
classification and worker joining are compatibility gates. New private headers
are installed only if required transitively by an existing public interface.
The manifest retains `codec/dc_prediction.h`, `codec/dc_quantization.h`,
`codestream/dc_prediction.h`, `codestream/dc_group.h`, `codestream/headers.h`
and `codestream/encoder.h` at their original paths. Only implementation sources
and the formerly private predictor header move; installed consumers need no
new Modular include or new library component.

## Local baseline setup

The Windows qualification uses Microsoft's MSVC 19.37 packages downloaded from
the installed Visual Studio catalog into `build/modular-phase1/toolchain`.
Payload hashes are checked against the catalog and retained in
`toolchain-manifest.json`. The installed Windows SDK supplies UCRT/SDK headers
and libraries. GJXL's unmodified storage-toolchain probe must pass.

`build/modular-phase1/baseline-src` is a detached worktree at the baseline commit.
Its libjxl directory references the pinned source already in `third_party/`;
the reference revision is verified independently. Build, capture, test and timing
artifacts stay in `build/modular-phase1/`; the final report records their hashes
and the actual coverage achieved. No hardware or test result is inferred merely
from this protocol.
