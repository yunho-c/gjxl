# Native context-map compression

GJXL now searches richer encodings for entropy context maps and block context
maps. On the retained 65-image effort-4 cohort, this reduces mean BD-rate by
**0.403%** relative to the completed DC integer-mapping search (`e8abc34`). All
455 output files are smaller and decode to exactly the same linear-RGB float
pixels. The frontend and histogram budgets retain their existing policies.

## Encoding and integration

The native encoder compares the existing raw-ID Prefix representation with
legal fixed-width maps and raw or move-to-front maps coded with Prefix or ANS.
Both entropy coders also have a distance-one RLE alternative. RLE uses separate
literal/length and distance contexts, a singleton distance histogram, minimum
copy length three and bounded copy chunks. The new candidates use HybridUint
`(2,0,1)` for map values and `(0,0,0)` for lengths. The legacy candidate retains
its existing integer-mapping search.

Every candidate is serialized, including map flags, LZ77 fields, integer
configurations, histograms, payload and ANS state. Only a strictly smaller
complete map replaces the incumbent. This guarantees that a selected map
does not exceed its legacy representation. RLE token selection uses a bounded
cost heuristic; the search does not claim a globally minimal representation
or try arbitrary backreferences. File-level model decisions can change when
the map cost changes.

Entropy model construction eagerly stores the selected map bytes and exact
source map. Model costing and later emission reuse them. Const writers do not
mutate the cache; if a caller edits a public `EntropyCode` map, an exact input
comparison detects the stale cache and the writer computes a temporary map.
All-zero maps retain the three-bit shortcut. Managed storage bounds cover the
cache, simultaneous candidate writers, transformed map, token buffer and
Prefix/ANS scratch. Search failure preserves its output.

This applies to the shared native writer, including CPU and Metal workflows
at every effort. AC caps remain 32/64 and the DC cap remains 32. No frontend,
quantization, predictor, smoothing or reconstruction policy changes are
introduced. A byte-target controller can choose a different distance because
the encoded size changes. The production encoder has no libjxl runtime
dependency; the optional decoder oracle links libjxl into one test only.

## Rate and reconstruction qualification

The study reuses the original seven distances for each of 65 images. Fresh
baseline encodes reproduce all 455 pinned `e8abc34` codestream hashes. Each
candidate passes an independent `djxl` decode and exact linear-sRGB PFM hash
comparison before the existing measured quality score is reused.

| Reference, SSIMULACRA2 75–85 | PCHIP mean BD-rate | Akima mean BD-rate |
|---|---:|---:|
| Native DC integer-mapping search, `e8abc34` | −0.403451% | −0.403113% |
| Saved same-frame libjxl writer | −0.033650% | −0.034137% |
| Stock libjxl effort 4 | −0.407711% | −0.355449% |

These are arithmetic means of per-image BD percentages with complete coverage
and no extrapolation. All 65 curves improve against the native baseline;
36/65 beat the saved libjxl writer and 47/65 beat stock libjxl e4 with PCHIP.
The libjxl-writer comparison is computed directly from the two rate curves,
with audited codestream hashes and exact reconstruction identities across
the intervening DC-search and context-map changes. It is effectively a tie
on this cohort. Compared with that writer, 295 files are smaller, 159 larger
and one equal; a mean result does not establish per-image parity.

This is the original investigative cohort, not held-out data. Its large-image
subset consists of three scenes at three resolutions. The result supports
context-map compression as a concrete improvement; it does not establish
parity at every quality, effort or content distribution.

## Original timing observation and subsequent qualification

The separate timing run uses the 12 declared diagnostic images at their
original Q80 distances, eight participating CPU threads, four alternating
process pairs, two warmups and five measured complete public encode calls per
process. All 96 processes reproduce their corresponding rate-run codestream
hashes; 480 measured calls include final output publication.

The median of the 12 per-image paired latency ratios is **1.0085×**, or +0.85%
time. Per-image medians range from 0.921× to 1.080×. These observations are
provisional: another checkout's codec work appears in 7/34 sampled process
snapshots, and compiler processes appear in 4/34. The AC-powered Apple M4 Pro
was idle of other codec processes at the initial check, but concurrent work
started during collection. These data do not qualify isolated latency or
establish a speedup on the faster-looking inputs.

The separate [September 15 latency qualification](context-map-latency.md) now
passes its declared environment checks: 96 accepted pairs, 192 processes and
960 measured calls. It observes +0.59% median paired time, with per-image
changes from −0.70% to +3.51%. Ten contaminated pairs are excluded and retained.
The report records the user-authorized temporary `suggestd` suspension, its
verified resumption and the limits of this local warm-call comparison.

## Correctness and regression checks

The 351 context-map fixtures cover all seven representation families: simple,
raw Prefix, MTF Prefix, raw ANS, MTF ANS, RLE Prefix and RLE ANS. They cover up
to 256 histogram IDs, long runs exceeding the decoder window, exact consumed
bit counts and unaligned writes. All pass the pinned libjxl decoder oracle.
The same fixtures check deterministic selection, the legacy size fallback,
actual managed backing against declared bounds, stale-cache detection and
allocation-failure atomicity. Prepared-cache reuse performs no allocation.

Whole-encoder checks include 28 baseline/candidate comparisons covering CPU
and Metal efforts 1–10, ordinary DC rounding, high density, maximum compression
and maximum-error control. Every pair has identical decoded pixels. Eleven
additional paired golden fixtures were independently decoded before updating
their byte hashes. Entropy/serializer storage tests include failure injection
and the additional retained cache storage; the installed-header consumer
also passes.

The final Release suite passes **138/139 CTest checks**. The sole failure,
`quantization_pipeline`, also reproduces with the untouched baseline binary:
actual score `0.24919039011001587` versus pinned `0.24914586544036865`. That
unrelated golden score is left unchanged. All 1,573 qualification commands
succeed; source, binary, helper and input hashes pass the final audit.

## Reproduction and artifacts

[`tools/context_map/qualify.py`](../tools/context_map/qualify.py) and its
[instructions](../tools/context_map/README.md) provide the opt-in rate, timing
and saved-data report workflow. The run freezes production sources, including
new files, source archive, baseline/candidate binaries and Metal libraries,
input hashes, decoder, score ledger and helper versions. Changed sources
require a new output directory.

Local evidence is retained in `build/context-map-qualification-20260914/`:
manifest, source snapshot, codestreams, raw samples, command ledger, analysis,
audit, scope checks, direct tail comparison, process snapshots and environment
audit. Golden comparisons are in `build/context-map-golden-audit/`; native and
reference test logs are under `build/context-map-*.log`. Large artifacts remain
local. [Committed results](context-map-compression-results.json) retain the
per-image rate/timing summaries, validation counts and provenance hashes.
