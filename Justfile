set shell := ["zsh", "-cu"]

build_dir := "build"

# Configure and build all project targets.
build:
    cmake -S . -B "{{ build_dir }}" -DGJXL_BUILD_BENCHMARKS=ON
    cmake --build "{{ build_dir }}"

# Compare all Metal DCT implementations.
benchmark blocks="65536" iterations="200": build
    "{{ build_dir }}/gjxl_dct_benchmark" "{{ blocks }}" "{{ iterations }}"
