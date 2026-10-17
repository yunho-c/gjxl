# gjxl

`gjxl` is a **GPU-accelerated JPEG XL encoder** with 2-30x speedup vs. multithreaded CPU at equal-or-better compression rates. 

[Paper](assets/gjxl-paper-v1.pdf) | Blog post (coming soon)

> **Abstract**: JPEG XL offers state-of-the-art compression rate and versatility. However, this comes at the cost of high computational complexity, especially for encoding. High-resolution images, such as those shot on modern smartphones, can take over 10 seconds to encode at high effort, which may complicate widespread adoption of JPEG XL. Based on the observation that much of libjxl’s expensive computations are parallelizable, we empirically demonstrate the usefulness of GPUs for JPEG XL by developing gjxl, a GPU-accelerated encoder that offloads perceptual analysis workloads to the GPU while keeping entropy/bitstream workloads on the CPU. **gjxl achieves 2.8× speedup vs. libjxl at default effort** while simultaneously producing smaller files via a more aggressive use of compute-intensive features. gjxl is permissively licensed and available at [https://github.com/yunho-c/gjxl](https://github.com/yunho-c/gjxl).

> gjxl is still in early development. Please let us know in [issues](https://github.com/yunho-c/gjxl/issues) if you have any thoughts, feedback, or requests to share!

## Overview

**Performance**

![Speed-Compression Performance](assets/speed-compression-performance.svg)

(Note: BD-rate describes file size differences at equivalent visual quality.)



**Backends**

- ✅ CUDA (NVIDIA)
- ✅ Metal (Apple)
- 🔄 WebGPU (cross-platform; work in progress)
- ❌ ROCm (not planned; no AMD GPU on hand)
- ❌ OpenCL, Vulkan (not planned)



**Limitations**

- Lossless encoding (Modular mode) is in development.
- Transparency, animation, and container metadata are not supported yet.
- Lossless JPEG transcoding is not supported (yet?).
- gjxl does not include a standalone decoder (yet?).

If you are impacted by these limitations, please feel free to let us know.



**Et cetera**

- gjxl includes an encoding CLI (`gjxl`), C and C++ interfaces, and safe Rust bindings.
- Want to try gjxl? Check out [OIMG](https://github.com/yunho-c/OIMG), a graphical image optimization program with gjxl built in.



### Q&amp;A

<details>
<summary>What is JPEG XL? How does JPEG XL work?</summary>

JPEG XL is a modern image codec that typically achieves 30-60% smaller files compared to JPEG and 30-50% smaller files compared to PNG (lossless).

![Overview of the JPEG XL codec architecture, including VarDCT and Modular encoding, JPEG reconstruction, and decoding](assets/jpeg-xl-codec-architecture.svg)

*Figure 6 from [The JPEG XL Image Coding System: History, Features, Coding Tools, Design Rationale, and Future](https://arxiv.org/abs/2506.05987v2), by Sneyers et al. (2025), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

</details>


<details>
<summary>What's the point of this? When will I ever need this?</summary>

From a research standpoint, this demonstrates for the first time that GPU-based JPEG XL codec implementation is a viable idea. We believe that more dramatic speedup would be possible in the future, as demonstrated in prior GPU acceleration literature on JPEG and JPEG2000.

From a practical standpoint, gjxl is useful when converting a bunch of files into JPEG XL for archival purposes, since it's faster, especially when processing images in batch.

</details>


<details>
<summary>Did you use LLM assistance?</summary>

Yes. You might have noticed from suboptimal code quality. This project serves primarily as a research platform rather than a safe, stable, production-ready library. We plan to migrate gjxl to Rust and write from scratch, after finalizing optimizations and design.

</details>


<details>
<summary>Does gjxl use more or less energy/battery compared to libjxl?</summary>

Likely more, although we'd have to experiment to be sure.

</details>

## License

gjxl is licensed under [Apache-2.0](LICENSE). Third-party dependencies retain their own licenses. We heavily derive from  libjxl and libjxl-tiny: the entire encoder design (aside from GPU-centric reformulation) belong to libjxl maintainers. 

---

## Details

**Table of Contents**

- [Theory of operation](#theory-of-operation)
- [Encode an image](#encode-an-image)
- [Use as a library](#use-as-a-library)
- [Development](#development)
- [Documentation](#documentation)

## Theory of operation

JPEG XL's encoding pipeline is complex, yet most computations expose data parallelism (i.e., existence of many independently computable work units), which GPUs are well-suited for. `gjxl` offloads (to GPU) roughly the initial two-thirds of the encoding pipeline—which we collectively refer to as the *perceptual analysis pipeline*—consisting of the following stages (ordered as they operate in the encoder):

- Image preparation
  - XYB color transform
  - Adaptive filtering (gabor-like filter, edge-preserving filter)
  - Chroma-from-luma (CfL)
- AC strategy search (i.e., variable DCT block partitioning)
  - Forward/inverse DCT
  - Rate-distortion proxy cost
  - Optimal partition search (via greedy merge heuristic)
- Adaptive quantization
  - Trial quantization/reconstruction (inverse DCT, CfL, adaptive filtering, XYB conversion)
  - Perceptual refinement using Butteraugli perceptual error metric
- Tokenization
  - Zero-counting and prefix sum for token offsets
  - Emission of token values and entropy contexts
  - GPU AC tokenization overlaps with CPU DC tokenization and entropy-model preparation

Remaining stages (entropy coding and bitstream assembly) are comparatively serial and remain on the CPU.

## Encode an image

```sh
python3 tools/encode_image.py --encoder build/release/gjxl_encode \
  input.png output.jxl -- --distance 1.0 --effort 7
```

### Quality and effort

`--distance` is a positive Butteraugli target: smaller values request higher quality.

![libjxl quality-to-distance mapping: quality 90 corresponds to distance 1.0](assets/libjxl_distance_quality_colorbar.svg)

`--effort` controls the amount of search and refinement; the default is `7`.


| Effort | Transform search                                         | Adaptive-quantization updates |
| ------ | -------------------------------------------------------- | -----------------------------: |
| 1–4    | Fixed 8×8 DCT                                            | 0                             |
| 5–6    | Mixed transforms                                         | 1                             |
| 7      | Mixed transforms                                         | 2                             |
| 8      | Mixed transforms, more compression search                | 3                             |
| 9–10   | Mixed transforms; denser large-transform placement at 10 | 4                             |


To search for a target file size or bitrate, replace `--distance` with one of:

```sh
build/release/gjxl_encode --target-bytes 250000 input.pfm output.jxl
build/release/gjxl_encode --target-bpp 1.0 input.pfm output.jxl
```

These searches can perform multiple complete encodes. The default is up to 12 attempts with a 0.5% size tolerance. See [rate control](docs/rate-control.md) for details.

### Choose a backend

```sh
build/release/gjxl_encode --backend cpu --distance 1.0 input.pfm cpu.jxl
build/release/gjxl_encode --backend metal --distance 1.0 input.pfm metal.jxl
# In a CUDA-enabled build:
build/cuda/gjxl_encode --backend cuda --distance 1.0 input.pfm cuda.jxl
```

Default automatic target-size and maximum-error requests stay on CPU. → okay this is a problem and should be fixed

## Use as a library

### C and C++

Install the libraries, headers, CLI, and CMake package from a complete build:

```sh
cmake --install build/release --prefix "$HOME/.local"
```

Point your application's `CMAKE_PREFIX_PATH` at the install prefix, then link
the C interface:

```cmake
find_package(gjxl CONFIG REQUIRED COMPONENTS c)
target_link_libraries(my_program PRIVATE gjxl::c)
```

[`gjxl/gjxl.h`](include/gjxl/gjxl.h) provides reusable contexts and encoding from packed 8-bit sRGB RGB or RGBA. Initialize option structs with  `gjxl_*_options_init` functions, call `gjxl_encode`, and
release its output with `gjxl_buffer_free`. Check each result and use
`gjxl_get_last_error` for diagnostics.

C callers can link `gjxl::codestream` and use
[`EncodeLinearRgbVarDctCodestream`](src/codestream/workflow.h) with planar
floating-point linear RGB, or the [batch workflow](src/codestream/batch_workflow.h).
See the [installed interface](docs/installed-interface.md) for components and public headers.

### Rust

The repository contains the safe `gjxl` crate and raw `gjxl-sys` bindings:

```rust
fn encode_rgb(width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>, gjxl::Error> {
    let context = gjxl::Context::new(gjxl::Backend::Auto)?;
    let image = gjxl::ImageView::rgb8(width, height, width as usize * 3, pixels)?;
    context.encode(&image, gjxl::EncoderOptions {
        distance: 1.0,
        effort: 7,
        ..Default::default()
    })
}
```

Keep a context around for repeated encodes. Native builds require CMake/toolchain setup plus Rust and libclang for binding generation. The default
`native` feature builds from the surrounding checkout; packaged consumers must
set `GJXL_SOURCE_DIR` to a matching source checkout. Enable the `cuda` feature
to build CUDA support on Linux or Windows. Cross-compilation is not currently
supported. See the [Rust README](rust/README.md) for details.

## Development

Build and run the test suite in a separate directory:

```sh
cmake -S . -B build/test -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON
cmake --build build/test --parallel
ctest --test-dir build/test --output-on-failure
```

To check generated streams with the pinned independent libjxl decoder:

```sh
cmake --build build/test --target codestream-conformance
```

The conformance target builds the decoder from the recursive submodule checkout.
`GJXL_ENABLE_LIBJXL_REFERENCE=ON` additionally enables libjxl reference tests;
`GJXL_BUILD_BENCHMARKS=ON` builds the benchmark tools. Rust tests run with
`cargo test --manifest-path rust/Cargo.toml --workspace`.

## Documentation

- [Resident execution](docs/resident-execution.md): GPU residency and workflow design.
- [Managed resources](docs/resident-resources.md): memory admission and execution domains.
- [Image batch benchmarking](docs/image-batch-benchmark.md): throughput measurement.
- [Metal profiling](docs/metal-resident-profiling.md) and [CUDA profiling](docs/cuda-profiling.md).
- [Justfile](Justfile): build, encode, conformance, and benchmark recipes.

Other files in docs are development artifacts and will be cleaned... sometime.

