# CUDA exact arithmetic during integration

The integrated prediction-aware DC policy exposed an exact-coefficient mismatch
on a photograph after three AQ updates. The CPU and CUDA coefficient code agreed,
but small differences in GPU reconstruction and perceptual feedback eventually
changed the next raw quantizer decision. The legacy DC control preserved the
pre-integration CUDA behavior.

Exact evaluation now uses GPU kernels that follow the scalar CPU arithmetic:

- The inverse DCT accumulates both passes in double precision, without implicit
  multiply-add contraction. Its cosine basis is copied from the CPU implementation
  so the host math library defines the same constants on both paths.
- Gaborish pairs horizontal/vertical neighbors and rounds products before adding.
- Inverse color conversion receives the host's float cube-root bias. MSVC's
  result differs by one representable float from the previous CUDA constant.
  Cubing/subtraction uses explicit rounding; the color matrix retains explicit FMA.
- Butteraugli Gaussian filters use the CPU's normalized paired taps, four partial
  accumulators, mirrored five-tap boundaries and truncated wider boundaries.
  Exact evaluation uses materialized blur/split passes with existing dead scratch.
- Block L16 feedback sums pixels in CPU row order, one CUDA lane per transform.

The ordinary resident kernels, fused paths and compiler flags retain their
previous arithmetic. There is no new CPU reconstruction or metric fallback.
The metric still has a small numerical tolerance; the acceptance checks separately
require identical coefficient decisions, codestream bytes and reconstructed RGB.

The evaluator owns 1,344 doubles (10,752 bytes plus arena alignment) of device
basis storage. Its existing persistent arena and shared admission recipe both
include this buffer. Three preparation copies upload the cached host basis;
there is no new global device allocation, host vector or per-evaluation upload.
Butteraugli retains its existing 25 working planes and multiscale storage recipe.

Verification on RTX 3060 Laptop / SM86, CUDA 11.8, MSVC 19.37:

- `cuda_exact_arithmetic`: all seven supported DCT shapes, signed/impulse/DC and
  magnitude controls in 21 guarded batches; 41 CPU byte/pixel pipeline pairs over
  tiny, narrow, odd and textured images, legacy/new DC and zero/two/four AQ updates.
- Existing exact-AQ tests now require identical direct reconstructed pixels with
  mixed transform shapes, all three EPF passes, custom filter weights and four
  DC policies. Existing maximum-error and failure-atomicity checks also pass.
- Four photographic fixtures (flower, keong, riaphotographs, bliznaca), zero through
  four AQ updates: 20/20 codestream, raw-quantizer and reconstructed-RGB matches.
  Maximum observed block-feedback difference is 3.6e-7. The original keong
  discrepancy is included; fixture identities and logs remain in local evidence.
- Public CLI verification covers the same four photographs at efforts 1, 4, 8
  and 10: all 16 CPU/CUDA pairs have identical codestream hashes. All 32 outputs
  decode with pinned libjxl and yield finite, pair-identical Butteraugli scores.
- All 156 configured tests pass. The first full run passed 154 tests; two compiler
  subprocess checks failed because its shell lacked the MSVC include/library
  environment. Both pass after loading the configured toolchain (90 seconds).
  The initial full run took 745 seconds; no codec failure required a source fix.
- The six focused AQ/storage/Butteraugli/filter tests pass. Exact-arithmetic
  Compute Sanitizer memcheck reports zero errors with a child completion marker;
  that check is included in the explicit CUDA qualification workflow.

A complete-public-call latency check uses four alternating process pairs per
case, 17 calls per process, and medians after dropping the first two samples.
The control substitutes the immediately preceding `38d5fc5` exact/Butteraugli
objects and header into the candidate archive, retaining its existing profiling
instrumentation. Its output hashes match the frozen `6c90d83` control. Across
32 processes (544 calls), the median paired candidate/control ratios are:

| Input | Exact, effort 8 | Resident, effort 7 |
| --- | ---: | ---: |
| Flower | 1.036 | 0.984 |
| Keong | 1.035 | 1.004 |

The corrected exact path costs about 3.5% here. Resident bytes are unchanged;
these samples show no material sustained resident regression. This is a
single-device photo check, not a general speed claim. An earlier 48-process
screen against `6c90d83` also covered exact effort 4; its 3.1% resident-keong
increase did not reproduce against the immediate-parent control. Logs, hashes,
source identities and all cold/warm samples remain in integration evidence.

Run the checked-in arithmetic test with an additional photographic PFM using
`gjxl_cuda_exact_arithmetic_test --photo IMAGE.pfm`. The default test only needs
tracked test data. `--quick REPORT.txt` runs the synthetic/DCT cases and writes
its completion marker only after all checks succeed.

Broader full-suite, performance and cross-device qualification are recorded in
[cuda-integration-progress.md](cuda-integration-progress.md). Automatic CUDA
selection remains opt-in while the integration PR is draft.
