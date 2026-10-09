# Accepted CUDA catch-up on current main

This integration carries three previously accepted changes onto `96cce2d`:

| Source commit | Integrated behavior |
| --- | --- |
| `8956a872d59c937d120bddc2abd226d957a2a227` | Fixed 32-row direct short filters for ordinary CUDA Butteraugli preparation. |
| `a1cd7c16de09326710ffb230f48f050be6f93651` | Reclaim completed idle cache entries after an actual CUDA allocation OOM, then retry once. |
| `f698cdc603435d806d99f875cbbbc8bb4c7f6589` | Include admission-time backend selection in the workflow profile. |

The source patches are applied selectively. The experimental CUDA AC tokenizer,
device-only completed-frame prototype, and unrelated branch changes are excluded.
The current effort policies and CUDA EPF inverse-sigma initialization fix remain
in place. No public ABI or codec policy changes are introduced.

## Short filters

Normal CUDA Butteraugli preparation uses the qualified fixed 32-row direct
high-X/Y, B-medium and ultra-X/Y filters. Tap order and rounded division are
preserved. They reuse the existing 25-plane arena with disjoint transient inputs
and outputs, reducing the affected preparation stage from eleven launches to
seven without adding persistent device allocation. CPU-order/exact-coefficient
preparation retains the legacy implementation.

No runtime selector is required. `GJXL_CUDA_SHORT_FILTER_TEST_SCHEDULES` defaults
to OFF; it enables only the alternate 16/64-row schedules for differential tests.
The default fixtures compare separate and fused passes bit-for-bit, with padding,
guards, malformed requests, repeated A/B/A use, and tall image shapes.

The original September qualification found modest warm complete-encode gains on
the RTX 3060 Laptop. Those historical measurements precede the current E1–E5
policies and are not a new performance claim. Unscored zero-refinement encodes
do not run the perceptual metric, so this integration does not predict a gain
for those low-effort paper rows.

## Allocation recovery

After `cudaErrorMemoryAllocation`, the allocator frees completed idle buffers
from its private pool and retries the requested allocation once. Other errors
and explicit failure injection retain their existing behavior. Live buffers,
allocation-domain accounting and their cache-return leases remain valid.

This handles idle cached allocations of incompatible sizes; it cannot make
arbitrarily large simultaneously live allocations fit. Successful allocations
do not add a trim or retry. The regression fixture verifies idle eviction,
failure-atomic output/statistics, preservation of live sentinel data and leases,
and subsequent exact-size reuse.

## Profile accounting

The private admission planner accepts an optional accumulator for backend
selection time. It adds the duration only after successful planning; failure
preserves both the plan and the accumulator. Later selection adds to this
duration instead of replacing it. Ordinary calls with a null accumulator read
no additional clocks. Planning outside an encode's boundary remains outside its
profile; this is attribution correction, not an encoding speed optimization.

## Integration validation

Fresh validation uses the Windows RTX 3060 Laptop, SM86, CUDA 11.8, MSVC 14.37,
and a separate Release build. Commands, original files, imported patches,
binary identities, native test results, sanitizer logs and public encode/decode
comparisons are retained under `build/cuda-accepted-main-20261008/`.

The 2026-10-08 integration checks completed successfully:

- All 164 native CTests passed: 115 CPU/common and 49 CUDA, with no skips.
- Short filters passed memcheck, racecheck, synccheck and initcheck. The allocator
  passed memcheck and initcheck. All six runs reported zero errors or hazards.
- Thirty-eight before/after settings produced 76 public encodes with identical
  codestream bytes and reported summaries/scores. Coverage includes efforts
  1–10, an odd 257x129 fixture, Kodak, 3 MP CLIC and a 12 MP photograph, with
  scoring enabled and separate unscored E1–E5 controls.
- All 24 unique output streams were independently decoded with the study's
  pinned libjxl decoder; every decoded pixel was finite.
- The short-filter kernel, header and fixture match the accepted commit's
  source. All seven multiline PowerShell blocks in the CUDA workflow parse.

`ready.json` records source/executable hashes and the completed checks. The
default build keeps compact AC and alternate short-filter schedules disabled.
These checks do not establish a new speedup or constitute Metal, Linux, Rust,
or remote-CI execution.

The manual CUDA qualification workflow now runs the two short-filter CTests,
all four short-filter sanitizer modes, and allocator memcheck/initcheck. Only
the allocator fixture suppresses sanitizer API-error reporting because it
deliberately requests an unsatisfiable allocation; its assertions still check
the returned error, eviction, live ownership and recovery. Device memory errors
remain checked. This integration does not add an experimental-tokenizer CI lane.

The completed paper study and its frozen source/binaries are preserved. These
changes do not rewrite its results or silently substitute new measurements.
