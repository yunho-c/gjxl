set shell := ["zsh", "-cu"]

build_dir := "build"

# Configure and build all project targets.
build:
    cmake -S . -B "{{ build_dir }}" -DGJXL_BUILD_BENCHMARKS=ON
    cmake --build "{{ build_dir }}"

# Compare all Metal DCT implementations.
benchmark blocks="65536" iterations="200": build
    "{{ build_dir }}/gjxl_dct_benchmark" "{{ blocks }}" "{{ iterations }}"

# Measure the CPU quantization and adaptive-quantization baseline.
quantization-benchmark workload="all" samples="5" warmups="3":
    cmake -S . -B "{{ build_dir }}/release" -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON
    cmake --build "{{ build_dir }}/release" --target gjxl_quantization_benchmark -j
    "{{ build_dir }}/release/gjxl_quantization_benchmark" --workload "{{ workload }}" --samples "{{ samples }}" --warmups "{{ warmups }}"
