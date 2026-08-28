set shell := ["zsh", "-cu"]

build_dir := "build"

# Configure and build all project targets.
build:
    cmake -S . -B "{{ build_dir }}" -G Ninja -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build "{{ build_dir }}"

# Quickly decode a representative corpus with the installed libjxl tools.
codestream-smoke:
    cmake -S . -B "{{ build_dir }}" -G Ninja -DGJXL_BUILD_TESTS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build "{{ build_dir }}" --target codestream-smoke

# Build the pinned libjxl decoder and run the complete conformance corpus.
codestream-conformance:
    cmake -S . -B "{{ build_dir }}" -G Ninja -DGJXL_BUILD_TESTS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build "{{ build_dir }}" --target codestream-conformance

# Encode a PFM directly or prepare a normal still image with ImageMagick.
[positional-arguments]
encode input output distance="1.0" *args:
    cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=OFF -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF
    cmake --build build/release --target gjxl_encode
    python3 tools/encode_image.py --encoder build/release/gjxl_encode "{{ input }}" "{{ output }}" -- --distance "{{ distance }}" "${@:4}"

# Probe a PFM across increasing Butteraugli targets and emit CSV.
[positional-arguments]
rate-control-probe input targets="0.5,0.75,1.0,1.2,1.5,2.0,3.0,4.0" *args:
    cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=OFF -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF
    cmake --build build/release --target gjxl_rate_control_probe
    build/release/gjxl_rate_control_probe --targets "{{ targets }}" "${@:3}" "{{ input }}"

# Measure bounded-memory PFM loading in a Release build.
pfm-benchmark input samples="9" warmups="2":
    cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build build/release --target gjxl_pfm_benchmark -j
    build/release/gjxl_pfm_benchmark --input "{{ input }}" --samples "{{ samples }}" --warmups "{{ warmups }}"

# Compare all Metal DCT implementations.
dct-benchmark blocks="65536" iterations="200": build
    "{{ build_dir }}/gjxl_dct_benchmark" "{{ blocks }}" "{{ iterations }}"

# Compare CPU and Metal batched AC candidate evaluation in a Release build.
ac-strategy-benchmark dct8_equivalents="4096" samples="12" quant_norm_source="host-quant":
    cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build build/release --target gjxl_ac_strategy_benchmark
    build/release/gjxl_ac_strategy_benchmark "{{ dct8_equivalents }}" "{{ samples }}" "{{ quant_norm_source }}"

# Compare complete CPU and staged-GPU AC searches in alternating order.
ac-strategy-search-benchmark samples="12":
    cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build build/release --target gjxl_ac_strategy_search_benchmark
    build/release/gjxl_ac_strategy_search_benchmark "{{ samples }}"

# Measure CPU and Metal quantization workflows with alternating phase order.
aq-benchmark workload="all" implementation="simd" samples="5" warmups="3" gpu_aq="exact-coefficients":
    cmake -S . -B "{{ build_dir }}/release" -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON
    cmake --build "{{ build_dir }}/release" --target gjxl_quantization_benchmark -j
    "{{ build_dir }}/release/gjxl_quantization_benchmark" --workload "{{ workload }}" --implementation "{{ implementation }}" --gpu-aq "{{ gpu_aq }}" --samples "{{ samples }}" --warmups "{{ warmups }}"

# Measure only the public CPU/Metal encoder boundary with phase profiles.
encode-benchmark workload="padded_1080p" implementation="simd" samples="3" warmups="1" gpu_aq="exact-coefficients":
    cmake -S . -B "{{ build_dir }}/release" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build "{{ build_dir }}/release" --target gjxl_quantization_benchmark -j
    "{{ build_dir }}/release/gjxl_quantization_benchmark" --scope public-workflow --workload "{{ workload }}" --implementation "{{ implementation }}" --gpu-aq "{{ gpu_aq }}" --samples "{{ samples }}" --warmups "{{ warmups }}"

# Measure repeated Metal public encodes after one CPU/Metal validation pair.
metal-encode-benchmark workload="padded_4k" implementation="simd" samples="7" warmups="2" gpu_aq="fully-resident":
    cmake -S . -B "{{ build_dir }}/release" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build "{{ build_dir }}/release" --target gjxl_quantization_benchmark -j
    "{{ build_dir }}/release/gjxl_quantization_benchmark" --scope metal-public-workflow --workload "{{ workload }}" --implementation "{{ implementation }}" --gpu-aq "{{ gpu_aq }}" --samples "{{ samples }}" --warmups "{{ warmups }}"

# Compare warm sequential and bounded-concurrency multi-image throughput.
image-batch-benchmark workload="all" batch_sizes="1,2,4,8" samples="3" warmups="1" backend="metal" gpu_aq="maximum-throughput":
    cmake -S . -B "{{ build_dir }}/release" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build "{{ build_dir }}/release" --target gjxl_image_batch_benchmark -j
    "{{ build_dir }}/release/gjxl_image_batch_benchmark" --workload "{{ workload }}" --batch-sizes "{{ batch_sizes }}" --samples "{{ samples }}" --warmups "{{ warmups }}" --backend "{{ backend }}" --metal-aq "{{ gpu_aq }}"

# Measure the CPU coefficient-decision boundary without the complete AQ loop.
coefficient-benchmark workload="padded_1080p" samples="9" warmups="2":
    cmake -S . -B "{{ build_dir }}/release" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build "{{ build_dir }}/release" --target gjxl_quantization_benchmark -j
    "{{ build_dir }}/release/gjxl_quantization_benchmark" --scope coefficient-coding --workload "{{ workload }}" --samples "{{ samples }}" --warmups "{{ warmups }}"

# Measure native CPU and prepared Metal Butteraugli paths.
butteraugli-metal-benchmark workload="all" samples="15" warmups="3":
    cmake -S . -B "{{ build_dir }}/release" -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON
    cmake --build "{{ build_dir }}/release" --target gjxl_metal_butteraugli_benchmark -j
    "{{ build_dir }}/release/gjxl_metal_butteraugli_benchmark" --workload "{{ workload }}" --samples "{{ samples }}" --warmups "{{ warmups }}"
