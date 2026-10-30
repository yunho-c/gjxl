# Prior-art correction to gjxl paper v1

October 30, 2026

## Scope and version

The [v1 paper](../assets/gjxl-paper-v1.pdf), generated on October 9, 2026 and uploaded in [commit 64c57ae](https://github.com/yunho-c/gjxl/commit/64c57aebde86b033bb459618986144a17a86c3fb) on October 12, contains an unsupported claim in its Conclusion that gjxl is the first GPU-accelerated JPEG XL encoder. We withdraw that priority claim.

The editable DCC draft's related-work discussion, conclusion and prior-art references have been corrected and verified. The hosted v1 PDF is preserved in its original form; this notice does not represent a regenerated or replaced PDF. See [issue #51](https://github.com/yunho-c/gjxl/issues/51) for the correction discussion.

## Missing prior art

- **[jxl-gpu](https://github.com/imazen/jxl-gpu)**, by Imazen, pursues GPU acceleration with a numerical-parity goal against libjxl-derived reference routines. Its documented implementation and tests do not demonstrate universal byte equivalence with libjxl's encoded output.
- **[cujpegxl](https://github.com/m-pilia/cujpegxl)**, by Martino Pilia, is a CUDA JPEG XL encoder that prioritizes throughput while accepting compression-efficiency tradeoffs.
- **[Pilia's technical comparison, October 24, 2026](https://martinopilia.com/posts/2026/10/24/cujpegxl-entropy-improvement.html)** discusses these implementations and their differing objectives and entropy-coding tradeoffs. This is an author-published technical report.

These projects must be credited in discussions of GPU-accelerated JPEG XL encoding. We thank Martino Pilia for identifying the omission.

## Contribution and performance scope

gjxl's contribution is its implementation and evaluation of GPU-oriented VarDCT encoding: resident intermediate state in the perceptual-analysis pipeline; CUDA and Metal backends; batched transform-candidate evaluation; kernel fusion, tiling and intermediate reuse; parallel AC tokenization; and CPU/GPU task overlap. Entropy coding and bitstream assembly remain on the CPU.

The paper's measured encoding-time and rate-distortion results concern gjxl and libjxl under its stated hardware, datasets, thread count and effort policies. The revised effort ladder enables some coding tools at lower effort numbers, so matching the nominal effort number does not imply identical encoder policies.

Those measurements do not establish a controlled performance ranking against jxl-gpu or cujpegxl. Comparisons among the projects require matched hardware, input representation, batching, timing boundaries and reconstructed quality. This correction adds prior-art attribution and removes the unsupported priority claim; it does not introduce new benchmark results.

