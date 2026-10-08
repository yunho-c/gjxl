# P2.4: exact palette and scalar squeeze

P2.4 adds prescribed exact palettes and reversible squeeze to the private scalar
Modular encoder. Public API/CLI integration, parallel CPU scheduling and GPU
kernels remain later milestones. The identity/gradient default is unchanged;
transform search is part of the existing explicit `search=true` option.

## Architecture and supported profile

`codec/modular/transform/transform.h` defines a bounded descriptor sequence and
channel shapes. `transform.cpp` validates ranges, applies metadata changes without
allocations, and composes forward operations or inverses in reverse order.
`palette.cpp` owns exact palette construction and inversion; `squeeze.cpp` owns
wide scalar arithmetic and inversion. `operations.h` is a composition-only
interface whose caller supplies validated shapes and unpublished output storage.
The optional existing RCT runs first, affecting only the first three color planes.
These modules have no entropy, stream-placement, search, GPU or libjxl dependency.

Each sequence has at most eight palette/squeeze descriptors and 64 resulting
channels. A descriptor addresses ordinary channels using the channel order at
that point in the sequence. Transforming metadata channels is deliberately
unsupported. Validation rejects invalid kinds/counts/ranges before allocation.

An exact palette combines one to four adjacent, equally shaped channels. Its
capacity is prescribed from 1 through 256; tuples are ordered lexicographically
using signed sample values. Unused capacity is zero-filled, deterministically.
There are no implicit colors, delta entries, lossy approximations or alpha
premultiplication. Exceeding capacity fails without publishing a replacement.
The new palette plane is prepended to the metadata prefix; the index plane
replaces the first selected channel and the remaining selected planes disappear.
Existing metadata and remaining image channels retain their relative order.
Native metadata descriptors use the metadata role with zero shifts; the pinned
codec represents the same special metadata placement with negative shifts.

Each squeeze descriptor explicitly chooses horizontal/vertical splitting,
in-place/appended residual ordering, and one through 19 adjacent channels.
Selected axes must have at least two samples and shifts below eight; empty
residual channels and implicit default squeeze programs are outside this subset.
The average plane uses the ceiling half-length, the residual uses the floor
half-length, and both inherit the incremented shift. Wide intermediates and
checked signed32 stores reject unsupported sample ranges. Forward/inverse
arithmetic retains odd tails and matches the pinned scalar rules, including
negative values and truncation toward zero in inverse reconstruction.

`ModularEncoderFrame::Prepare` accepts the resolved coding policy and exposes an
immutable transformed image. The workflow derives stream placement from that
image. Metadata remains global; ordinary channels are assigned afresh according
to shape and shifts, including DC-group streams. Tokenization counts actual
plane areas, including palette metadata, instead of source pixels times input
channels. Source image metadata remains the original gray/RGB/RGBA 8/16-bit
profile, and transformed streams conservatively clear the signed16-buffer flag.

`stream_encoder.cpp` serializes the ordered transform descriptors in the global
stream header. Group headers carry no additional transforms. The existing
VarDCT stream-header helper and whole-image Modular default syntax are retained.
Output is still a single regular, final, single-pass frame.

## Search and resource contracts

Search policy stays in `codestream/modular/search.*` and `workflow.cpp`.
After the existing identity/RCT and learned-tree candidates, search tries:

1. An exact full-pixel palette when an allocation-free probe finds at most 256
   distinct tuples. The probe reads the declared sample byte order and stops
   immediately at the 257th tuple.
2. One horizontal squeeze, when width permits.
3. One vertical squeeze, when height permits.
4. Horizontal followed by vertical squeeze, when both axes permit.

Squeeze search uses RCT 6 for color and identity for gray. These four additional
choices use the gradient leaf; prescribed transforms can use the existing
weighted predictor and multi-leaf policies. Search evaluates at most 11 complete
color files or seven gray files. Complete byte size, including metadata, tree,
models, transforms, sections and padding, selects the winner; earlier candidates
win ties. Identity/RCT-only output therefore remains available. A palette that
exceeds the search limit is ineligible; allocation or encoding failures are
terminal and never silently skipped. Prescribed policies are never overridden.

The allocation-free storage plan applies every shape change before computing
stream, token and entropy bounds. It accounts for all intermediate image stages
conservatively, including the borrowed original and replacement overlap, palette
planes, maximum predictor width, global sample payload and nonempty DC sections.
Search reserves the largest candidate envelope plus a retained winner. Existing
managed output remains charged separately. Forward operations, inverses, frame
preparation and final publication move replacements into the output only after
success; input samples remain unchanged. All execution remains serial with one
admitted CPU participant.

## Qualification

See [the machine-readable record](modular-p2.4-results.json) for exact results,
commands, artifact hashes and retained baseline failures. The environment is the
same audited Windows x64 MSVC build under ARM64 emulation used for P2.3; the
independent reference is libjxl `e8ff09762481785938d8e4e01333ed3917571161`.

Palette qualification ran first: 168 prescribed cases and 336 complete files.
The expanded transform oracle covers 313 cases and 626 complete prefix/ANS files.
It builds palette expectations with an independent ordered dictionary and pinned
metadata application, invokes pinned `FwdSqueeze` and `FwdRct`, compares every
intermediate plane and shape, verifies native inverse recovery, and independently
decodes complete files into the original integer samples. It does not require
libjxl's palette-search heuristics to choose the same representation.

Coverage includes all six source formats, one-color/small/256-entry palettes,
transparent colored RGBA pixels, remaining-channel and multiple-metadata ordering,
palette channel counts 1/2/3/4, both squeeze directions/orderings, narrow and odd
shapes, group/DC boundaries, eight-step sequences, the 64-channel and 19-channel-range limits, nonzero channel starts,
RCT/palette/squeeze combinations in both palette/squeeze orders, and weighted
prediction with a channel-property tree. Workflow tests cover invalid descriptors,
257-color rejection, signed32 squeeze overflow, atomic failure and recovery,
source immutability, exact-budget admission, nested ownership and exhaustive
managed allocation failure injection for palette, squeeze, combinations and search.

| Check | Result |
| --- | --- |
| Transform stage cases / complete prefix and ANS files | 313 / 626 exact |
| Existing policy stage cases / complete files | 799 / 1,646 exact |
| Frozen prescribed P2.3 policy files | 1,598 unchanged |
| Search comparisons against P2.3 | All 48 no larger; 17 smaller |
| Managed allocation failure injection | 51,050 positions across 44 cases |
| Frozen default Modular / VarDCT artifacts | 2,802 / 208 unchanged |
| Pinned VarDCT full decoder conformance | 23/23 pass |
| Reference-disabled build, install and installed workflow consumer | Pass |
| Installed C11/C++20/C++23 consumers and relocation | Pass |
| Full repository CTest | 121/130 pass; no lost P2.3 passes |
| Native include boundary / installed package | 60-file closure; no private headers or libjxl exports |

The broad run took 404.08 seconds. The same nine baseline failures/not-run tests
remain: two Windows contract-test build failures, five Butteraugli reference/
differential targets, missing generated Butteraugli goldens, and the known
255x256 DC-processing oracle mismatch. Eight corresponding broad-build failures
also remain. These are unresolved failures, not passes. The earlier seven-test
focused Modular run passed; all seven also passed in the final broad run.

Raw evidence is retained under `build/modular-p2.4/`, with the 626 transform
artifacts in `build/modular-phase1/candidate/modular-transform/`. Build
`gjxl_modular_transform_test` with the configured pinned reference and run it
through CTest's `modular_transform` entry; the target is test-only and never
installed. The `palette` executable argument after its artifact directory runs
only the initial palette milestone.

GPU execution, native ARM64, other platforms and sanitizers remain unqualified.
These synthetic fixtures do not establish corpus compression ratio or throughput.

## P2.5 handoff

Preserve the serial private path as the CPU oracle. Add admitted parallel stream
encoding with fixed model data and canonical assembly before exposing the public
C++/C/Rust and CLI surfaces described in the architecture plan. Keep palette and
squeeze selection separate from their scalar implementations. Empty residuals,
metadata transforms, larger palettes, deeper transform search and responsive
passes require explicit capability and resource work; they are not prerequisites
for P2.5's initial bounded profile.
