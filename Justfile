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

# Encode a linear-RGB PFM with the standalone native CPU workflow.
encode input output distance="1.0":
    cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=OFF -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF
    cmake --build build/release --target gjxl_encode
    build/release/gjxl_encode --distance "{{ distance }}" "{{ input }}" "{{ output }}"

# Compare all Metal DCT implementations.
benchmark blocks="65536" iterations="200": build
    "{{ build_dir }}/gjxl_dct_benchmark" "{{ blocks }}" "{{ iterations }}"

# Compare CPU and Metal batched AC candidate evaluation in a Release build.
ac-strategy-benchmark dct8_equivalents="4096" samples="12":
    cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build build/release --target gjxl_ac_strategy_benchmark
    build/release/gjxl_ac_strategy_benchmark "{{ dct8_equivalents }}" "{{ samples }}"

# Compare complete CPU and staged-GPU AC searches in alternating order.
ac-strategy-search-benchmark samples="12":
    cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON -DHWY_ENABLE_TESTS=OFF
    cmake --build build/release --target gjxl_ac_strategy_search_benchmark
    build/release/gjxl_ac_strategy_search_benchmark "{{ samples }}"
