# P2.3: scalar predictors, RCT and bounded global policy

This milestone extends the private Modular CPU encoder from P2.2 with prescribed
prediction/transform/tree choices and opt-in deterministic compression search.
The default remains identity transforms and the global gradient leaf. Public APIs,
CLI integration and GPU kernels remain later milestones.

## Prescribed capability contract

`ModularEncodingOptions::coding` supplies `ModularCodingPolicy`: a bounded decision
tree, weighted predictor parameters and an RCT type. `search` defaults to false.
A non-default prescribed policy cannot be combined with search; rejection happens
before managed allocation. Gray/RGB/unassociated RGBA at 8/16 bits retain P2.2's
byte order, stride, source-space integer and alpha-preservation contracts.

The scalar path implements all 14 wire predictor IDs (0..13): zero, left, top,
average0, select, gradient, weighted, top-right, top-left, left-left, and averages
1 through 4. Encoder-only predictor IDs are rejected. Stream/channel boundaries
reset predictors; row boundaries reset the previous local-gradient property.
Integer averages use the decoder's truncation toward zero. RCT shifts use signed
arithmetic shift, which rounds negative odd values downward.

Weighted parameters contain seven five-bit coefficients and four four-bit maximum
weights, with the original Phase 1 defaults. Both default and explicit headers are
supported. The predictor retains its five managed two-row arrays. Its default
behavior remains identical for embedded VarDCT callers. Invalid parameter values
are rejected before writing. Intermediate arithmetic is wide; retained signed
error and unsigned predictor-error ranges are checked before state publication.

RCT 0 means no transform. Types 1..41 implement the specified channel permutations
and reversible arithmetic; type 6 is YCoCg. Forward and inverse scalar kernels are
qualified against pinned libjxl and exact source reconstruction. RCT applies only
to color planes 0..2 and is rejected for grayscale. Alpha is never transformed.
The descriptor appears once in the global Modular stream header; group streams
carry the weighted parameters but no repeated RCT. Source color metadata stays
sRGB/gray sRGB and unassociated alpha is preserved.

Trees contain at most 31 nodes/16 leaves. Properties 0..15 include channel, stream
ID, local coordinates, neighbor values/differences and weighted error. Previous-
channel properties and local trees/models are unsupported in this first policy.
Nodes use explicit children: left means `property > split`, right means `<=`.
Validation rejects invalid predictors/properties, unreachable nodes, cycles,
shared children and contradictory property ranges along a path. Both branches of
a split must remain reachable within the inherited signed-32-bit property range.

Leaves have a signed-32-bit offset and a multiplier in `[1, INT32_MAX]`. Encoding
requires `(sample - prediction - offset)` to divide exactly by the multiplier and
the quotient to fit signed 32 bits. Otherwise the request fails atomically; it
never approximates or silently changes a prescribed leaf. Serialization traverses
breadth-first and assigns context IDs in leaf visitation order, independently of
input node numbering. Predictor properties and tree tokens are checked against
the pinned implementation.

All non-default prescribed policies conservatively clear the signed-16-buffer-
sufficient declaration. The P2.2 default still sets it for 8-bit identity/gradient
input. RCT source-derived values lie within `[-65535,65535]`; non-weighted predictor
and property calculations use `int64_t`. Weighted-state and residual conversions
remain checked, including for custom parameter sets.

## Deterministic search

`search=true` uses a fixed, bounded policy with no mutable global state or random
seed. It first encodes the original identity/gradient baseline with the requested
entropy coder, then considers RCTs in order `0, 6, 9` (only 0 for gray).

For each transform, training visits canonical streams, channels and raster samples.
All pixels update predictor state. It records samples at ordinal multiples of
`ceil(total_samples / 4096)`, keeping at most 4,096 records. Each record stores all
16 properties and the residual score for all 14 predictors. The score is
`1 + bit_width(PackSigned(residual))`, an integer training proxy, not a claimed
entropy or complete-file size estimate.

The learner proposes a single best predictor and optionally one split with two
leaves. Predictor ties prefer gradient, then ascending predictor ID. Split search
visits properties 0..15, then thresholds zero and the sample mean (integer division
toward zero). Empty branches are excluded. It selects each branch's best predictor
and charges a fixed 32-score split penalty; equal scores retain the earlier choice.
Learned offsets and multipliers remain zero and one. Prescribed trees expose the
wider offset/multiplier and topology capabilities without claiming deeper learning.

Candidates are encoded completely, single leaf before split, skipping exact policy
duplicates. The winner is the smallest complete byte vector, including image/frame
headers, RCT/weighted/tree headers, entropy models, TOC and padding. Baseline and
earlier candidates win equal sizes. At most seven complete candidates are encoded
for color input and three for gray. Resource or encoding failures abort atomically,
rather than silently skipping a failed candidate. Successful search cannot exceed
the corresponding default baseline file size.

This deliberately small first learner leaves deeper trees, additional thresholds,
weighted-parameter search, previous-channel properties, local trees/models and
broader candidate sets as explicit future extensions. Palette/squeeze remain P2.4.

## Architecture and resource accounting

The codec layer separates weighted state (`prediction.h`), scalar neighborhood
prediction (`scalar_prediction.h`), tree representation/properties/validation
(`tree.h`), bounded learning (`tree_learning.*`), and RCT (`transform/rct.h`).
`coding.h` is only the resolved policy value and its validation. Frame preparation
owns integer loading and applying the selected color transform. The codestream layer owns tree tokenization, entropy contexts, stream headers,
canonical training-sample collection, complete-file candidate selection and storage
planning. The learner consumes property/cost records without entropy or stream-plan
dependencies. Stream-header emission receives only weighted parameters and the RCT
identifier, rather than the full tree or encoder policy. Production code uses no
libjxl routines.
The existing Phase 1 stream-header subset remains available to VarDCT callers.

Storage planning scales tree tokens, contexts and potential entropy clusters with
the validated tree. Both prefix and ANS models use at most 16 contexts/clusters,
within the shared 8-bit context-map and 32/64-cluster limits. Signed-packed
residuals can use the complete uint32 range; shared hybrid-uint/token emission
bounds already cover it. Singleton/no-LZ77 behavior is unchanged. General trees,
including empty contexts and nontrivial leaf ordering, decode independently.

Preparation accounts for one weighted state at a time. Search additionally
accounts for the bounded managed sample array and a retained complete winner while
training/encoding the next candidate. Training releases frame/layout/sample backing
before candidate encoding. Candidate encoding retains P2.1's phase lifetimes.
Memory admission and CPU participation precede work and extend through publication;
nested candidate calls reuse the same execution domain and participant. Existing
caller-owned managed output stays charged separately during replacement.

## Qualification

The environment remains the audited Release x64 MSVC 19.37.32826/STL 202305 build
on Windows ARM64 under emulation, with Windows SDK 10.0.26100.0. The independent
reference is pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161`, built with
x64 clang-cl. See the [machine-readable record](modular-p2.3-results.json) for exact
commands, hashes, retained failures and measured candidate sizes/timings.

The new policy oracle compares every scalar prediction, all properties, weighted
state effects, RCT output, scalar inverse, breadth-first tree tokens, leaf contexts
and residual tokens. It uses libjxl's `PredictLearnAll`, weighted state, `FwdRct`
and `TokenizeTree`, independently of native calculations. Complete files are then
decoded into source-space integers. Tests include all six formats, 1x1, narrow/odd
sizes, group and DC-group boundaries, negative transformed samples, alpha-zero
colors, custom/zero/max weighted parameters, all predictors/RCTs/properties,
nontrivial child ordering, offsets/multipliers, and a 31-node tree.

| Check | Result |
| --- | --- |
| Prescribed stage comparisons | 799 exact |
| Complete prescribed/search files | 1,646 exact |
| Search comparisons | 48 deterministic, no larger than complete baseline |
| Managed allocation failure injection | 38,689 positions across 34 cases |
| Frozen P2.2 default Modular files | 2,802 unchanged |
| Frozen VarDCT artifacts | 208 unchanged |
| Pinned VarDCT decoder conformance | 23/23 pass |
| Reference-disabled native build and installed workflow consumer | Pass |
| Installed C11 and C++20/C++23 consumers and relocation | Pass |
| Complete CTest run | 120/129 pass; all P2.2 passes retained |
| Final focused Modular suite after module cleanup | 6/6 pass; all 1,646 policy artifacts unchanged |
| Native Modular include closure | 55 files; no VarDCT-policy, GPU or libjxl headers |

The broad run took 381.84 seconds with four parallel tests and a 600-second timeout.
The same nine baseline failures/not-run tests remain: two Windows contract-test
build failures, five Butteraugli reference/differential targets, missing generated
Butteraugli goldens, and the DC-processing oracle's previously reproduced 255x256
mismatch (channel 2, x=8, y=2, pattern 3; reference -5850, native -5852). The broad
build retains eight known failed targets. These are unresolved failures, not passes.
After that run, eight extreme offset/multiplier cases and the final module-boundary
cleanup were qualified with the complete focused Modular suite and a rebuilt
reference-disabled package/consumer; frozen artifact comparisons were repeated.

Each search comparison records complete baseline/selected byte counts and elapsed
whole-call time. These synthetic, single-run timings are qualification evidence,
not a corpus-level speed or compression-ratio claim. Search performs multiple
encodes and can be substantially slower than the default path.

Raw evidence is under `build/modular-p2.3/`. Policy artifacts are retained under
the candidate build's `modular-policy` directory. P2.2's 2,802 default artifacts and
Phase 1's 208 VarDCT artifacts use frozen expected hashes, without refreshing them.
The reference-disabled installed workflow consumer exercises both prescribed and
searched native paths with matching private source headers.

Build `gjxl_modular_policy_test` alongside the existing Modular targets; run
`ctest --test-dir build/modular-phase1/candidate -R '^modular_' --output-on-failure --timeout 600`
from the configured compiler environment. The record contains full build/install,
artifact comparison, installed-consumer and broad regression commands.

GPU execution, native ARM64, other platforms and sanitizers remain unqualified.
There is no unrestricted level-5, corpus compression-ratio or performance claim.
