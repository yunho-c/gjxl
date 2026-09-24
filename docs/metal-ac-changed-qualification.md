# Changed-arithmetic Metal AC default

The Metal candidate path adopts the qualified `changed-mag-safe-loss-selected`
bundle on the existing Apple GPU family 9 / 32-lane SIMD route. Five candidate
shapes (16×8, 16×16, 32×16, 16×32 and 32×32) use FP32 radix-2 DCTs; 8×8 and
8×16 retain SIMD matrix transforms. All seven use the qualified safe magnitude
addition and dimension-specialized quantization norm. The 16×8, 16×16 and
16×32 shapes fold the first loss-reduction levels in registers with explicit
addition ordering. No FP16 arithmetic is introduced.

This follows the [exact-output AC integration](metal-ac-exact-qualification.md)
at revision `81a675c2904e63522b505d42b157cc05880861c5`. It changes arithmetic and
can change candidate decisions, compressed bytes and decoded pixels. It is not
an exact-output optimization or a guarantee of non-regressing quality.

The factored kernels have distinct names, so an older external shader library
uses the existing grouped/split fallbacks rather than receiving an incompatible
threadgroup width. The per-shape pipeline availability, SIMD width, thread-count
and memory checks remain in place. Split fallback arithmetic, public interfaces
and automatic CPU/Metal backend selection are unchanged. Use `--backend metal`
to exercise this path at settings where automatic selection chooses CPU.

| Shape | Threads before → after | Static threadgroup bytes before → after |
|---|---:|---:|
| 8×8 | 96 → 96 | 1,408 → 1,408 |
| 16×8 | 192 → 96 | 7,424 → 2,304 |
| 8×16 | 96 → 96 | 3,584 → 3,584 |
| 16×16 | 192 → 96 | 5,632 → 4,608 |
| 32×16 | 384 → 192 | 17,408 → 9,216 |
| 16×32 | 192 → 192 | 14,336 → 9,216 |
| 32×32 | 384 → 384 | 22,528 → 18,432 |

## Retained performance and quality qualification

The qualification compared against the exact-output production default above
on an M4 Pro with 48 GB, **AC power** and Low Power Mode disabled. The selected
15 images comprise eight CLIC images, four Kodak images, and three Unsplash
scenes at 12/24/48 MP. Efforts 5/7/10 × distances 0.1/1.9/3 give 135 settings.
Two independent A/A and A/B cohorts per setting, three warmups and eight
alternating measured pairs produced 540 comparisons, 4,320 pairs and 8,640
measured encode calls. No environmental rejection occurred.

Geometric-mean reduction in warmed complete-encode time across the 15 images:

| Effort | Distance 0.1 | Distance 1.9 | Distance 3 |
|---:|---:|---:|---:|
| 5 | 3.45% | 3.99% | 4.07% |
| 7 | 2.77% | 3.43% | 3.51% |
| 10 | 2.84% | 3.67% | 3.59% |

Overall reduction was **3.48%**, with an aggregate A/A time change of +0.037%.
All 135 paired medians improved; 123 gains exceeded the observed control range
in both cohorts, and 12 overlapped it. These are descriptive comparisons, not
confidence intervals. Timing covers warmed public encode calls, excluding input
loading, pipeline creation, initial capacity priming, output checks and returned
object destruction. It does not qualify cold starts, the full 65-image corpus,
other devices, battery power or unrelated backend paths.

132/135 photo settings preserved both codestream bytes and decoded pixels.
The three changes were all the CLIC image
`11f2b039b293758398b1a7a8afa64bb2` at distance 0.1:

| Effort | File bytes | Size change | Butteraugli change | SSIMULACRA2 change |
|---:|---:|---:|---:|---:|
| 5 | −571 | −0.032934% | 0 | −0.032904 |
| 7 | +5 | +0.000278% | 0 | +0.006050 |
| 10 | −5 | −0.000275% | 0 | −0.015905 |

Butteraugli was unchanged for every photo setting. Linear RGB MSE improved
slightly in the three changed cases. The geometric-mean photo size change was
−0.000244%, effectively neutral in this sample.

Seven synthetic patterns × nine settings added 63 diagnostic quality cases.
54 were byte-identical; all nine impulse settings changed. The largest
Butteraugli regression was impulse effort 7/distance 1.9:
2.208891153 → 2.294495106 (**+3.875%**, worse), with SSIMULACRA2 −0.037835 and
six fewer bytes. At effort 10/distance 1.9, impulse Butteraugli improved
2.016477346 → 1.920653820 (−4.752%), with SSIMULACRA2 +0.345446 and size
+0.077212%. The largest impulse SSIMULACRA2 decrease was 0.072093 points.
This observed stress-case regression is part of the adoption tradeoff.

Quality used independent pinned `djxl` decoding to linear RGB float PFM,
native CPU Butteraugli at intensity 80, MSE/max RGB error and pinned
`fast-ssim2` revision `c3867954c7bec8a761951df9256354b305fd0cff`.
The retained 4,200 ordinary/Metal-validation kernel cases checked guards,
read-only inputs and no new finite-to-nonfinite results. Numerical differences
from the exact baseline were expected and measured.

## Production integration verification

The fresh Release build disables Metal profiling and contains no study hooks.

- The normal `gjxl_encode --backend metal --distance D --effort E INPUT.pfm
  OUTPUT.jxl`, with its embedded library and default AQ/thread settings,
  reproduces all **198 retained candidate codestreams byte for byte**: 135
  photo settings and 63 stress settings. This transfers the independently
  decoded size/quality results above exactly to the production output.
- All 2,100 ordinary direct-kernel cases match every compared word of the
  retained changed bundle. All seven dispatch and static-memory contracts match.
- The production Release library passes 2,100 Metal API/shader-validation guard,
  read-only and finite-result checks. Its instrumented arithmetic differs from
  the retained library, which was compiled with `-gline-tables-only
  -frecord-sources=yes`. Building the integrated source with those same debug
  flags restores exact numerical agreement in another 2,100 validation cases.
  Ordinary Release arithmetic is exact; instrumented cross-build arithmetic
  must not be used as the ordinary-output comparison. Both initial and matched
  diagnostic results are retained.
- Two independent timing cohorts on the CLIC portrait and 24 MP campus image
  at effort 7/distance 1.9 compare production with the retained candidate.
  Retained-versus-production paired median time changes ranged from -0.523% to +0.919%; unchanged-library controls ranged from -1.812% to +2.629%.
  Each uses two warmups and four alternating pairs; all calls repeat their own
  bytes, summaries and submission counts. This is an integration check, not a
  new full performance qualification.
- The final full CTest pass completed 154/155 tests with present fixtures;
  updating the profile test expectations for the renamed kernels then passed
  the remaining test on a focused rerun. The four tests
  needing the pre-existing deleted `testdata/codestream_sample.pfm` pass when
  replayed with a temporary HEAD copy outside the checkout: **159/159 checks**
  in total. The AC test retains its CPU-reference error bounds, exact
  wide-versus-compact comparison and same-path repeatability checks; it no
  longer asserts equality between deliberately different arithmetic paths.
- The current host also passes the complete AC test against the previous exact
  shader library under Metal API/shader validation, exercising the older-library
  fallback. The user's deleted fixture, all 35 pre-existing changed/untracked
  files and staged content are preserved.

Integrated source SHA-256: `d2cf44c7b111782c61ad1de3b0d93606649b5f5bad3adda3aeabdc691e9add61`.
Production metallib SHA-256: `e723dc40f52b5ababb6f07d08f3204930d339b9594dac73996f805d79fb9633c`.

The standalone `gjxl_metal_ac_candidate_probe` now uses production factored
kernel names and widths. Its bitwise comparison remains strict: comparing the
changed candidate against a split oracle can correctly report a difference.
The retained qualification harness separately measures numerical differences
and guarded behavior; its integration check requires exact agreement with the
retained changed bundle.

Retained source SHA-256:
`bd223d7e86ef976dfb244569bed974ff8683b5e656ba177de69961076f21d38c`.
Retained metallib SHA-256:
`3ecf4a8b6507e769c8ee02f4d8c7bb9d3f123e99bbf4ee9d0e73fc761ea3cd27`.

Local evidence locations (not build dependencies):

- `/Users/yunhocho/Documents/Codex/ac-changed-corpus-20260923`: frozen sources,
  libraries, protocol, raw timing controls, codestreams, independent quality
  results, per-image/stress CSV files and figures.
- `/Users/yunhocho/Documents/Codex/ac-changed-production-20260923`: preservation
  backup, production build, adapted validation harnesses and integration checks.
