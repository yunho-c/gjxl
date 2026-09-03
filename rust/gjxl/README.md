# gjxl

Safe Rust wrapper for GJXL's C interface.

The default `native` feature builds GJXL from the repository containing this
crate. Packaged consumers must set `GJXL_SOURCE_DIR` to a GJXL source checkout.
Native builds support macOS, Linux, and Windows; enable the `cuda` feature to
compile CUDA support on a non-macOS host with the CUDA toolkit installed.

```rust
let context = gjxl::Context::with_options(gjxl::ContextOptions {
    backend: gjxl::Backend::Auto,
    cpu_threads: Some(4),
})?;
let image = gjxl::ImageView::rgba8(width, height, width as usize * 4, &pixels)?;
let options = gjxl::EncoderOptions {
    distance: gjxl::distance_from_quality(90.0),
    effort: 7,
    compression_mode: gjxl::CompressionMode::Automatic,
};
let codestream = context.encode(&image, options)?;
# Ok::<(), gjxl::Error>(())
```

Set `compression_mode` to `gjxl::CompressionMode::Maximum` to opt into the
exhaustive entropy/codestream search. Automatic mode uses the ordinary
effort-aligned behavior.
