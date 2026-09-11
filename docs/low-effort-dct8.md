# Low-effort transform policy

Ordinary efforts 1–4 select DCT8 for every 8×8 base block without invoking
AC-strategy search. Effort 5 enables the existing mixed-transform search.
The adaptive-quantization update schedule stays unchanged:

| Effort | Transform selection | AQ updates |
| --- | --- | --- |
| 1–3 | Fixed DCT8 | 0 |
| 4 | Fixed DCT8 | 1 |
| 5–6 | Mixed-transform search | 1 |
| 7 | Mixed-transform search | 2 |
| 8–9 | Mixed-transform search | 3 |
| 10 | Mixed-transform search | 4 |

The high-density and maximum-error overrides retain their existing search
behavior. Target-byte and target-bpp retries follow the ordinary effort
policy. Maximum-compression controls the serializer independently and does
not re-enable AC search. The dedicated maximum-throughput path keeps its
existing policy. The low-level quantization pipeline defaults to mixed search;
the encoding workflow explicitly selects the effort policy.

The shared CPU/GPU orchestration builds a complete DCT8 grid before calling
adaptive quantization and never calls the AC provider in fixed mode. This
removes candidate preparation, transforms, scoring, readback, and CPU merging.
Resident preparations release retained AC-search scratch when switching to
fixed mode. CPU, resident Metal, and compatibility Metal admission plans omit
search-only storage and include overlap for replacement strategy grids.

Initial quantization, CfL, and inverse-Gaborish preparation remain. In
particular, the resident inverse-Gaborish image also feeds final coefficient
coding; removing it would change the encoded result. Effort 4 still performs
its AQ update, so its speed cannot be inferred directly from a zero-update
DCT8 experiment. Low-effort output bytes and rate-quality behavior change;
this change does not establish speed parity or a new matched-quality result.

## Resident search-data omission

Encoding with fixed DCT8 now prepares the Metal frontend without its initial
AC-search masks. The gradient and erosion kernels omit mask-only arithmetic
and stores; the pixel-mask convolution and validation dispatches are skipped.
The three mask planes use separate minimal argument bindings instead of
full-sized backing. Initial strategy-mask and CfL host readbacks are also
omitted. Initial quantization, device CfL, and inverse Gaborish still feed
final coefficient coding.

This applies to both zero-update efforts 1–3 and effort 4's AQ update. Full
initial-quantization diagnostics retain all masks and initial CfL outputs;
efforts using mixed-transform search retain their existing preparation. The
cached evaluator includes this choice in its compatibility check, so changing
between encoding, diagnostics, and mixed-transform search rebuilds the
appropriate preparation. Device and profile storage plans follow the same
policy. Shared host preparation bounds remain conservative.

## Validation

`low_effort_strategy_policy` checks the effort/override matrix, unchanged AQ
counts, provider bypass, mixed/fixed preparation reuse, and reduced storage
plans. It uses stub providers and performs no image encoding or GPU work.
The CPU, resident, and compatibility workflow storage tests support
`--plans-only` for their allocation-free plan matrices (and mock policy or
search-interval checks).

The workflow regression test also checks DCT8-only summaries on CPU and Metal
at efforts 1–4, includes effort 5, and retains the scored/unscored byte-parity
checks. The GPU pipeline regression covers switching mixed/fixed preparations,
zero candidate statistics, and scored/unscored fixed-policy byte parity.

Release compilation and the non-encoding checks passed: the new policy test,
4,480 CPU plan shapes, 2,240 resident plan shapes, and 2,800 compatibility plan
shapes, plus their mock-policy and search-bound checks. The resident plan test
now includes score/timing publication storage in its aggregate bound comparison;
without AC search, it cannot assume a strict saving from releasing search
storage before completion.

After encoder runs were enabled, the Release workflow, low-effort policy,
CPU/resident/compatibility storage-plan, and Metal quantization-pipeline tests
passed. The resident retry-lifetime assertion now expects no AC-search storage
for fixed DCT8 and retained search storage for mixed-transform preparations.
The three previously calibrated DCT8 probes also match the production e1
output bytes exactly. These checks do not establish corpus-wide rate-quality
behavior or speed parity.

The search-data omission tests compare initial quantization, final CfL, and
encoded bytes against the full preparation. They exercise diagnostic/encoding
cache transitions with zero and one AQ update, forbidden mask/CfL outputs,
and atomic numeric/readback failures. Metal API and shader validation also
pass on the affected workflows. This establishes output parity and less
search-only backing; measured complete-encode timings remain mixed.
