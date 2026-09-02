# Release notes

## Unreleased

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

See
[`entropy-behavior-alignment.md`](entropy-behavior-alignment.md) for source
provenance, exact behavior, conformance coverage, and retained performance and
size results.
