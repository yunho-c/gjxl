# Exact-output Metal AC default

The ordinary Metal candidate kernels now use the qualified `selected-specloss-all` bundle. The existing Apple GPU family 9, 32-lane SIMD, thread-count and threadgroup-memory checks select these kernels; optional-pipeline failure retains the existing fallback. Automatic CPU/Metal backend-selection policy still applies. Select `--backend metal` to exercise the Metal path at distances such as 0.1 where automatic selection uses CPU.

Five candidate shapes (8×8, 8×16, 16×16, 16×32 and 32×32) reuse their transform arena and halve magnitude scratch. The 16×8 and 32×16 shapes retain their previous scratch arrangement. All seven specialize loss indexing to their known dimensions. The original reduction tree and explicit FP32 magnitude rounding are preserved. In-place transforms share the existing DCT templates through a compile-time mode; barriers separate input reads from aliased stores, and inverse dispatch assertions require every participating SIMD group to own a row tile. Public API and kernel names are stable.

The retained qualification used main revision `715f033acc90495237562068a5f4e7db7cd3b024`, an M4 Pro with 48 GB on AC power, and 15 images (8 CLIC, 4 Kodak, and three Unsplash scenes at 12/24/48 MP). Efforts 5/7/10 × distances 0.1/1.9/3 give 135 settings. Two independent passes, A/A controls and A/B comparisons, three warmups and eight alternating timed pairs produced 540 comparisons and 8,640 timed encode calls. All 135 settings preserved bytes and encoding summaries across both passes.

Geometric-mean reduction in warmed complete-encode time across the 15 images:

| Effort | Distance 0.1 | Distance 1.9 | Distance 3 |
|---:|---:|---:|---:|
| 5 | 3.21% | 3.21% | 3.18% |
| 7 | 2.22% | 2.71% | 2.43% |
| 10 | 1.32% | 2.30% | 2.39% |

Unchanged-library aggregate controls ranged from −0.21% to +0.32%. Paired median time decreased in 131/135 settings; 111 settings showed gains beyond the observed control range in both passes. These are descriptive measurements, not statistical confidence bounds. Loading, pipeline creation, initial capacity priming, output checks and destruction of returned objects are outside the timer. Matching codestreams imply identical compression size and decoded output under the same decoder. This qualifies the selected corpus on the measured device, not all 65 source images, cold starts or all GPU families.

Production integration was checked with a fresh Release build, Metal profiling disabled, and no study hooks:

- 4,200 direct kernel cases (2,100 ordinary, 2,100 with Metal API/shader validation) match every compared word of the retained bundle, with guards intact. All seven threadgroup-memory and dispatch contracts also match.
- The normal `gjxl_encode --backend metal --distance D --effort E INPUT.pfm OUTPUT.jxl`, using its embedded library and default AQ/thread settings, reproduces all 135 qualified codestreams byte for byte.
- Two-pass timing spot checks on the selected CLIC portrait and 24 MP campus image at effort 7/distance 1.9 differ from the retained bundle by less than 0.5%; unchanged-library controls span −0.34% to +0.62%. This checks the integration, not a second complete performance sweep.
- The original CTest run passed 155/159 tests. Four fixture-dependent tests failed because the checkout already had `testdata/codestream_sample.pfm` deleted. All four passed when replayed with a temporary copy from HEAD outside the checkout. The user's deletion and other unrelated work were preserved.

Retained shader source SHA-256: `697719bfc6ba44f1aceb2da8d14685805d366dce8d0fe30e2419169535054c11`.
Integrated shader source SHA-256: `87a304302f12c6bcbb478473971885706f8ba679701be7c4c729aa14777ed8d1`.
Release metallib SHA-256: `14ddea78e8a27f2321a2d98dd4f4729acdd79e0a14f9b3be0396dc5a7f298672`.

The local evidence bundles are `/Users/yunhocho/Documents/Codex/ac-exact-corpus-20260923` (measurement plan, identities, per-image CSV, raw controls and codestreams) and `/Users/yunhocho/Documents/Codex/ac-exact-production-20260923` (production build, structural/CLI proofs, fixture replay, timing checks and preservation record). They are evidence locations, not build dependencies.
