# Release notes

## Unreleased

### Fully resident Metal encoding default

- Fully resident AQ is now the default Metal computation path for codestream
  encoding. Qualified automatic Butteraugli-target requests use it; forced
  Metal also uses it when no AQ mode is specified.
- Exact coefficients remain available as an explicit C++ and CLI
  reference/compatibility mode. The source-compatible full diagnostic GPU
  overloads continue to default to exact coefficients.
- Resident floating-point decisions may change the quant field, frame, and
  codestream bytes relative to CPU or exact mode. CPU fallback and operational
  failure atomicity are unchanged.
- Automatic resident target-byte and target-BPP searches remain entirely on
  CPU to avoid mixing CPU and resident rate curves within one search.
  Automatic maximum-error control also remains CPU-only. Forced Metal retains
  resident support for both policies.
- Resident effort levels 1-3 perform the sole terminal evaluation needed for
  their zero-update policy; higher efforts continue to omit the final
  diagnostic score unless requested.
- The C and Rust APIs gain no implementation-mode field and inherit the
  canonical automatic/default behavior.

### Entropy behavior alignment

- Balanced, effort-7-like entropy/codestream behavior is now the default for
  direct serializer calls and efforts 1-8. It commits one block-context map and
  coefficient-order representation, selects Prefix or ANS before model
  construction, and eliminates the former outer candidate tournament.
- Efforts 9-10 and the high-density compatibility override use an
  effort-9-like direct ANS policy with population-cost cluster refinement, the
  pinned 28-entry libjxl HybridUint search, and precise histogram selection.
- The former exhaustive serializer remains available through
  `VarDctCompressionMode::kMaximumCompression`, CLI and benchmark
  `--maximum-compression`, C API `GJXL_COMPRESSION_MAXIMUM`, and safe Rust
  `CompressionMode::Maximum`.
- `GJXLEncoderOptions` grows from 12 to 16 bytes. Its size-versioned ABI still
  accepts the former 12-byte layout and defaults the absent compression field
  to automatic behavior.
- Default codestream bytes intentionally change. The checked 17x13 balanced
  sample is 263 bytes with SHA-256
  `e4566239f5e15dd67a4716d26da662728c88ffcae19bdf93ff28c2b8df6c8504`.
  Maximum compression preserves the former 255-byte SHA-256
  `e5577ebf76a37bf56a93db61b2ccf1fc959292a3d13d6489baf2e7f5b6105558`.
- Raw workflow benchmark schema 14 records requested effort, compression mode,
  resolved entropy behavior, and HybridUint/histogram/alphabet-width candidate
  counts.
- The ordinary balanced serializer now reuses fixed ANS populations, skips
  redundant exact candidate measurement, parallelizes independent AC-group
  preparation through the bounded CPU budget, and uses sparse clustering
  distance calculations. In paired Release measurements this reduced
  serializer wall time from 92.692 to 27.327 ms at 1080p and from 148.195 to
  61.610 ms at 4K; maximum-compression selection remains exact and unchanged.
- The Opsin frontend uses pinned libjxl's vector-friendly cube-root refinement,
  with a four-lane NEON implementation on Apple Silicon and a tested scalar
  fallback. The transform enforces a `1e-6` scalar-reference error bound and
  does not add a platform framework dependency.
- A fresh matched-quality effort-7 comparison puts GJXL at 1.375x libjxl
  complete-encode time for the representative 1080p photo and 1.203x at 4K.
  Remaining serializer overhead is concentrated in coefficient tokenization
  and entropy construction; GJXL model/token emission and framing are already
  faster on both inputs.

See
[`entropy-behavior-alignment.md`](entropy-behavior-alignment.md) for source
provenance, exact behavior, conformance coverage, and retained performance and
size results.
