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

## Lossless Modular

`Context::encode_modular` preserves integer samples and unassociated alpha; it
uses CPU for automatic execution. This is independent of `encode` and its VarDCT
quality/distance controls.

```rust
let image = gjxl::ImageView::new(width, height, stride_bytes, &pixels,
                                gjxl::PixelFormat::Rgba16BeSrgb)?;
let encoded = context.encode_modular(&image, gjxl::ModularOptions {
    search: true,
    entropy: gjxl::ModularEntropy::Ans,
})?;
```

Use `Gray8Srgb`, `Rgb8Srgb`, `Rgba8Srgb`, or the `Gray16`, `Rgb16`, `Rgba16`
variants ending in `LeSrgb`/`BeSrgb`. Buffers are byte slices with explicit byte
order; padded rows and unaligned data are supported. Source colors are sRGB (or
its gray transfer function); no float conversion or alpha premultiplication is
performed. Bindgen includes the new sized C options and entry point automatically.
