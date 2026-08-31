# Rust bindings

GJXL provides two Rust crates:

- `gjxl-sys` owns raw bindings, native source builds, and static link metadata;
- `gjxl` owns validated image views, reusable contexts, buffer lifetime, and
  typed error translation.

Run their tests from the repository root:

```bash
cargo test --manifest-path rust/Cargo.toml --workspace
```

The crates currently support native macOS builds only. When used from this
repository they locate the source tree automatically. Packaged consumers set
`GJXL_SOURCE_DIR` to a GJXL checkout containing the matching C API.
