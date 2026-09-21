# Rust bindings

GJXL provides two Rust crates:

- `gjxl-sys` owns raw bindings, native source builds, and static link metadata;
- `gjxl` owns validated image views, reusable contexts, buffer lifetime, and
  typed error translation.

Run their tests from the repository root:

```bash
cargo test --manifest-path rust/Cargo.toml --workspace
```

Native builds support macOS, Linux, and Windows. macOS enables Metal; the other
platforms build the portable CPU backend by default. Enable the safe crate's
`cuda` feature to compile and link the CUDA backend on a machine with the CUDA
toolkit. When used from this repository the crates locate the source tree
automatically. Packaged consumers set `GJXL_SOURCE_DIR` to a GJXL checkout
containing the matching C API.
