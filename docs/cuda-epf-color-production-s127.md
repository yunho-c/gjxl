# Production final-EPF/color fusion (S127)

Starting revision: `0a9eb89`, branch `feat/cuda`; Windows, MSVC 14.37,
CUDA 11.8, RTX 3060 Laptop / sm86, 2026-09-08.

## Production change

The fully resident perceptual evaluator now converts the last EPF pass
directly to linear RGB. One EPF iteration ends with pass 1; two or three end
with pass 2. Earlier EPF passes, Gaborish, reconstruction, sigma preparation,
and maximum-error evaluation retain their existing dataflow. Maximum-error
reduction needs the final filtered XYB image; retaining that materialization
is a live consumer requirement, not a compatibility layer. Zero EPF iterations
still invoke standalone color conversion.

`EpfTiledKernel<Pass, ToLinear>` shares one EPF implementation between both
output policies. A shared device helper retains the color arithmetic and
numeric error handling. Active filtering sanitizes non-finite EPF output and
ORs `2u` into the error word before conversion; bypass sends the unmodified
XYB values to color conversion, whose non-finite output ORs `4u` into the
error word. Existing error bits remain set.
The halo, mirror rules, barrier placement, tile loop and bypass continuation
are unchanged. Tiles remain 32x32 with 256 threads.

The new internal launcher accepts final passes 1/2, validates matching extents,
consumed strides and non-null storage, and handles matching empty extents as
no-ops. The caller still validates allocations and nonaliasing. The separate
EPF output stride and color input stride are unused because the intermediate
does not exist. RGB uses its own output stride, including source-width-packed
output with coding-width-padded XYB input.

Each eligible evaluation removes one launch and three global plane writes
plus three reads: 24 logical bytes per source pixel, or 198,921,624 bytes for
3839x2159. This is eliminated boundary traffic, not a claim about all physical
DRAM transactions. Existing scratch allocations remain in place. No public
API, compact-coefficient default, scheduling policy, compatibility layer,
fast-math setting or power setting changes.

## Native-code qualification

The clean production build contains 213 GPU bodies, adding only two fused
specializations to S124's 211. Both fused bodies are instruction-identical to
the [S125](cuda-epf-color-fusion-s125.md) candidates used by
[S126](cuda-epf-color-integration-s126.md), after normalizing symbols. All three
ordinary EPF bodies remain instruction-identical, despite their new internal
template signatures and unused extra argument.

The standalone color helper refactor swaps some X/Y address/register operand
assignments and the corresponding subtraction operands. It preserves the
instruction count, opcode histogram, resource usage and arithmetic. It is
not claimed to be byte-identical. Overall 210 original bodies are unchanged.
The regression and old-library comparisons test the actual rebuilt code,
including zero-EPF and maximum-error consumers of standalone conversion.

| Final pass | Ordinary EPF registers | Fused registers | Shared bytes |
| --- | ---: | ---: | ---: |
| 1 | 40 | 55 | 15,552 |
| 2 | 38 | 40 | 13,872 |

Both fused bodies have zero stack/local memory; standalone conversion remains
20 registers with no shared/stack/local memory. Nine of ten GPU ELF modules
are byte-identical to the frozen S124 build. Substituting the audited new
module gives the exact module-hash multiset of eleven release/host-ASAN test
and encoder executables, including both dense and compact owners. This avoids
repeated full-executable disassembly without weakening module identity checks.

## Permanent regression and integration checks

The new `cuda_epf_color` CTest compares separate EPF/color against fusion
bitwise, using guarded planes and a nonblocking stream. It covers 24 shapes,
including one-row/one-column images, 32-pixel tile boundaries, odd sizes and
a 4096x3 strip; packed and padded RGB layouts; distinct input/intermediate/RGB
pitches; both final passes; two color scales; sixteen input/sigma patterns;
and three changed-input reuses. Patterns include signed zero, subnormals,
extreme finite values, infinities, NaNs, positive/NaN sigma, threshold-adjacent
bypass values, and alternating bypass rows.

Every check preserves inputs, sigma, row/allocation guards and seeded error
bits. The fused path leaves the poisoned XYB scratch untouched. Independent
error-bit assertions distinguish filtered non-finite values from bypassed
non-finite values. Invalid arguments and empty launches are also checked for
unchanged output/error storage. The existing EPF test retains its independent
CPU oracle; the new test does not replace it.

All 79 CTest tests pass in the freshly configured Release build, with no skips
or timeouts. The new test performs 18,432 pipeline executions in CTest and
another 18,432 under host ASAN; its smaller preflight performs 2,304. The
existing CUDA AQ test also passes under host ASAN. ASAN instruments the test
harness and resident owner, not every library or GPU instructions.

A diagnostic profile matrix links the same driver separately against frozen
S124 libraries and the rebuilt release/host-ASAN production libraries. Each
build evaluates 128 combinations: two contents, zero through three EPF
iterations, Gaborish on/off, ordinary/custom filter and intensity parameters,
perceptual/maximum-error mode, and zero/two AQ updates. Quant fields, block
maps, score histories, complete RGB planes, maximum-error fields and
serialization outcomes are recorded without structure padding. All records
match bitwise: 256 old/new AQ comparisons. Eight cases per build serialize;
120 match existing unsupported-profile writer rejections with empty output.
Thus only sixteen compared pairs are successful codestream serialization,
not all 256. Resident frame population checks pass in every case.

The S108 frozen-oracle campaign passes 960 ordinary encodes, 222 under host
ASAN and 48 under encoder memcheck: 1,230 exact encodes, plus fourteen matching
expected rejection cases. Coverage includes photographs, padded HD/4K,
quality/final-score options, byte and maximum-error search, extreme finite
ranges, compact width/overflow decisions, reused and independent contexts,
and public batch execution. Both dense and compact owners are rebuilt from
the actual production source; no diagnostic fusion selector remains.

All four encoder memcheck jobs report zero errors and leaks. The permanent
EPF/color test passes another 2,304 executions under each of memcheck,
racecheck, initcheck and synccheck, with zero reported errors/hazards. Including
CTest, host ASAN and the preflight, that is 48,384 guarded pipeline executions.
All eight CUDA sanitizer jobs pass.

## Performance interpretation

S127 promotes the native-identical fused candidates qualified in S126; it
does not repeat that timing campaign or claim new timing samples. S126's
actual in-encoder stage measurements save about 0.40 ms at HD and 1.61 ms at
odd 4K (about 29% at 4K), with all 24 individual candidate/control stage medians
favorable. Its uninstrumented whole-encode results are mixed, including one
slower 4K window. Those limits still apply: this is a verified local stage
improvement, not evidence of a universal whole-encode throughput gain or a
maxed-out encoder. S125's synthetic counter measurements are not relabeled
as counters collected from S127 production inputs.

## Reproduction and audit

Artifacts are under `U:/gjxl-cuda-diagnostics/s127`; driver sources are under
`build-cuda-ninja/profiles/s127_*`. The build uses CUDA 11.8, sm86, Release,
MSVC 14.37, Ninja, tests enabled, Metal/reference decoder disabled and compact
AC off. Separate diagnostic owners qualify compact AC and host ASAN. The
S108 corpus, specifications and frozen oracles are reused without regeneration.

The initial clean build failed only because the new test passed a
`string_view` directly to `runtime_error`. The original test source and failed
log are retained; explicit string construction fixed it, then the fresh build
directory was completed. A subsequent driver used the wrong success-footer
text for the existing AQ test. That executable returned zero with its correct
success footer and no ASAN report; `aq_marker_correction.json` records the
acceptance correction without overwriting the original record. The resumed
driver begins with the profile matrix, not a duplicate test run. Neither
incident was a GPU, firewall or elevation failure.

No timed GPU campaign runs in this turn. GPU correctness jobs are serialized;
CPU-only native-module extraction overlaps CTest. The final validator checks
job return codes and log hashes, explicitly listed expected rejections,
matrix identities, native module identities, qualification input hashes and
forty retained older runtime hashes. The source snapshot and artifact manifest
preserve the final evidence independently of later live-source edits.
