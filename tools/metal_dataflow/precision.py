#!/usr/bin/env python3
"""Reproducible, isolated Metal precision experiments; never rewrites production shaders."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/precision-study"
ENV = {k: v for k, v in os.environ.items()
       if not k.startswith(("GJXL_", "MTL_", "ASAN_", "UBSAN_"))}

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def run(name, command, extra=None, accepted=(0,)):
    command = [str(v) for v in command]
    record = OUT / "commands" / (name + ".json")
    log = record.with_suffix(".log")
    if record.exists():
        old = json.loads(record.read_text())
        if (old["command"] == command and old["environment"] == (extra or {}) and old["exit"] in accepted
                and old["log_sha256"] == sha(log)):
            return old
        raise RuntimeError(f"Existing command differs or failed: {record}")
    record.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env={**ENV, **(extra or {})},
                                stdout=stream, stderr=subprocess.STDOUT)
    value = dict(command=command, environment=extra or {}, start=start,
                 end=time.time(), exit=result.returncode, log=str(log),
                 log_sha256=sha(log))
    save(record, value)
    print(name, result.returncode, flush=True)
    if result.returncode not in accepted:
        raise RuntimeError(f"{name} failed: {log}")
    return value

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    identity = OUT / "identity.json"
    if identity.exists():
        raise RuntimeError("Baseline already prepared; use build or variants")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names += ["docs/metal-precision-study.md"]
    plane_probe = "benchmarks/metal_butteraugli_plane_probe.cpp"
    if (ROOT / plane_probe).exists():
        names.append(plane_probe)
    names += [str(p.relative_to(ROOT)) for pattern in
              ("benchmarks/metal_precision*.cpp", "tools/metal_dataflow/precision*.py")
              for p in ROOT.glob(pattern)]
    source = OUT / "src"
    sources = {}
    for name in sorted(set(n for n in names if n)):
        src, dst = ROOT / name, source / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            dst.symlink_to(src, target_is_directory=True)
        else:
            shutil.copy2(src, dst)
            sources[name] = sha(src)
    save(identity, dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        sources=sources, protocol=sha(ROOT / "docs/metal-precision-study.md"),
        submodules=subprocess.check_output(["git", "submodule", "status", "--recursive"], cwd=ROOT, text=True)))
    run("configure", ["cmake", "-S", source, "-B", OUT / "baseline", "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DGJXL_BUILD_TESTS=ON", "-DGJXL_BUILD_BENCHMARKS=ON",
        "-DGJXL_ENABLE_LIBJXL_REFERENCE=OFF", "-DGJXL_ENABLE_METAL_PROFILING=OFF"])

def build():
    baseline = OUT / "baseline"
    run("baseline-build", ["cmake", "--build", baseline, "-j", "8"])
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    libraries = ["codestream", "metal", "pfm_io", "gpu_butteraugli", "gpu_ops", "gpu", "codec"]
    run("probe-build", ["/usr/bin/c++", "-std=c++20", "-stdlib=libc++", "-O3", "-DNDEBUG",
        "-isysroot", sdk, "-I", OUT / "src/src", "-I", OUT / "src/include",
        OUT / "src/benchmarks/metal_precision_probe.cpp", "-o", baseline / "gjxl_metal_precision_probe",
        *[baseline / f"libgjxl_{n}.a" for n in libraries],
        "-framework", "Metal", "-framework", "Foundation", "-framework", "CoreGraphics"],
        {"SDKROOT": sdk})
    artifacts = [p for p in baseline.glob("gjxl_*") if p.is_file()]
    artifacts += list((baseline / "metal").glob("*.ir")) + [baseline / "metal/gjxl.metallib"]
    save(OUT / "baseline-artifacts.json", {str(p): sha(p) for p in artifacts})

def metric_probe():
    source = ROOT / "benchmarks/metal_precision_metric_probe.cpp"
    folder = OUT / "probes"
    folder.mkdir(parents=True, exist_ok=True)
    revision = sha(source)[:12]
    frozen = folder / ("metric-" + revision + ".cpp")
    shutil.copy2(source, frozen)
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    libraries = ["metal", "pfm_io", "gpu_butteraugli", "gpu_ops", "gpu", "codec"]
    binary = folder / ("metric-" + revision)
    run("metric-build-" + revision, ["/usr/bin/c++", "-std=c++20", "-stdlib=libc++", "-O3", "-DNDEBUG",
        "-isysroot", sdk, "-I", OUT / "src/src", "-I", OUT / "src/include", "-I", OUT / "src/tests",
        frozen, "-o", binary, *[OUT / "baseline" / f"libgjxl_{n}.a" for n in libraries],
        "-framework", "Metal", "-framework", "Foundation", "-framework", "CoreGraphics"], {"SDKROOT": sdk})
    save(folder / "metric.json", dict(binary=str(binary), sha256=sha(binary), source_sha256=sha(source)))

STRICT = ["-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise", "-ffp-contract=off"]
VARIANTS = {
    "ba_contract": ("butteraugli", [*STRICT[:2], "-ffp-contract=fast"]),
    "ba_fastfn": ("butteraugli", [STRICT[0], "-fmetal-math-fp32-functions=fast", STRICT[2]]),
    "ba_relaxed": ("butteraugli", ["-fmetal-math-mode=relaxed", "-fmetal-math-fp32-functions=fast"]),
    "ba_explicit_fma": ("butteraugli", STRICT),
    "epf_contract": ("aq_postprocess", [*STRICT[:2], "-ffp-contract=fast"]),
    "epf_fastfn": ("aq_postprocess", [STRICT[0], "-fmetal-math-fp32-functions=fast", STRICT[2]]),
    "reduce_fastfn": ("aq_reduction", [STRICT[0], "-fmetal-math-fp32-functions=fast", STRICT[2]]),
    "aq_contract": ("aq_reconstruction", [STRICT[0], "-ffp-contract=fast"]),
    "ba_half_mask": ("butteraugli", STRICT),
    "ba_half_blur_scratch": ("butteraugli", STRICT),
    "ac_half_forward": ("ac_strategy", []),
    "ac_half_all": ("ac_strategy", []),
    "dct_half_all": ("dct", []),
    "ac_half_storage": ("ac_strategy", []),
    "ba_relaxed_precise": ("butteraugli", ["-fmetal-math-mode=relaxed", "-fmetal-math-fp32-functions=precise"]),
    "ba_relaxed_nocontract": ("butteraugli", ["-fmetal-math-mode=relaxed", "-fmetal-math-fp32-functions=precise", "-ffp-contract=off"]),
    "ba_malta_reassociate": ("butteraugli", STRICT),
    "ba_filter_reassociate": ("butteraugli", STRICT),
    "ba_filter_nocontract": ("butteraugli", STRICT),
    "ba_selective_reassociate": ("butteraugli", STRICT),
    "ba_malta_reciprocal": ("butteraugli", STRICT),
    "ba_malta_fastdivide": ("butteraugli", STRICT),
    "ba_reciprocal": ("butteraugli", [*STRICT, "-freciprocal-math"]),
    "control": ("butteraugli", STRICT),
}

MIXED_MATRIX = """
// Experiment: round multiplicands to half, retain FP32 accumulation/results.
template <typename MatrixA, typename MatrixB>
inline void PrecisionMma(thread simdgroup_float8x8& result,
                         MatrixA a, MatrixB b,
                         simdgroup_float8x8 accumulator) {
  simdgroup_half8x8 ah, bh;
  ah.thread_elements() = __builtin_convertvector(a.thread_elements(), simdgroup_half8x8::storage_type);
  bh.thread_elements() = __builtin_convertvector(b.thread_elements(), simdgroup_half8x8::storage_type);
  simdgroup_multiply_accumulate(result, ah, bh, accumulator);
}
inline void PrecisionMultiply(thread simdgroup_float8x8& result,
                              simdgroup_float8x8 a, simdgroup_float8x8 b) {
  auto zero = make_filled_simdgroup_matrix<float, 8>(0.0f);
  PrecisionMma(result, a, b, zero);
}
"""

def transformed(name, code):
    if name in ("ba_malta_reciprocal", "ba_malta_fastdivide"):
        division = "1.0f / (norm + absolute)" if name == "ba_malta_reciprocal" else "fast::divide(1.0f, norm + absolute)"
        old = "  const float scaler = norm2_0_gt_1 / (norm + absolute);"
        assert code.count(old) == 1
        code = code.replace(old, f"  const float reciprocal = {division};\n  const float scaler = norm2_0_gt_1 * reciprocal;")
        old = "  const float scaler2 = norm2_0_lt_1 / (norm + absolute);"
        assert code.count(old) == 1
        return code.replace(old, "  const float scaler2 = norm2_0_lt_1 * reciprocal;")
    if name == "ba_selective_reassociate":
        return transformed("ba_malta_reassociate", transformed("ba_filter_nocontract", code))
    if name in ("ba_malta_reassociate", "ba_filter_reassociate", "ba_filter_nocontract"):
        start_marker, end_marker = (
            ("inline float malta_scale_value(", "inline float l2_asymmetric(")
            if name == "ba_malta_reassociate" else
            ("inline float convolve_transposed_vertical_value(", "inline float maximum_clamp("))
        start, end = code.index(start_marker), code.index(end_marker)
        contract = "off" if name == "ba_filter_nocontract" else "fast"
        code = (code[:start] + f"#pragma clang fp reassociate(on) contract({contract})\n" + code[start:end]
                + "#pragma clang fp reassociate(off) contract(off)\n" + code[end:])
    if name == "ba_explicit_fma":
        old = "  const float product = multiplier * value;\n  return product + addend;"
        assert code.count(old) == 1
        return code.replace(old, "  return fma(multiplier, value, addend);")
    if name == "ba_half_mask":
        start = code.index("kernel void gjxl_butteraugli_fuzzy_erosion_f32(")
        end = code.index("kernel void gjxl_butteraugli_masked_ac_f32(", start)
        part = code[start:end]
        assert part.count("device float* output") == 1
        part = part.replace("device float* output", "device half* output")
        code = code[:start] + part + code[end:]
        # Only eroded reference masks. Blurred reference/distorted masks remain
        # FP32 so their small differences are computed from unrounded inputs.
        assert code.count("device const float* mask ") == 4
        assert code.count("device const float* mask,") == 1
        code = code.replace("device const float* mask ", "device const half* mask ")
        code = code.replace("device const float* mask,", "device const half* mask,")
    if name == "ba_half_blur_scratch":
        start = code.index("kernel void gjxl_butteraugli_frequency_low_medium_tiled_f32(")
        end = code.index("#undef GJXL_LOW_MEDIUM_COARSENED", start)
        part = code[start:end].replace("threadgroup float*", "threadgroup half*")
        # All products and accumulations retain float operands. Only the
        # horizontal-pass output rounds on writing threadgroup storage.
        code = code[:start] + part + code[end:]
    if name in ("ac_half_forward", "ac_half_all", "dct_half_all", "ac_half_storage"):
        if name in ("ac_half_forward", "ac_half_storage"):
            start = code.index("inline void AcStrategyForwardSquareDct(")
            end = code.index("inline void AcStrategyInverseSquareDct(")
            part = code[start:end].replace("simdgroup_multiply_accumulate(", "PrecisionMma(")
            code = code[:start] + part + code[end:]
        else:
            code = code.replace("simdgroup_multiply_accumulate(", "PrecisionMma(")
            code = code.replace("simdgroup_multiply(", "PrecisionMultiply(")
        marker = "using namespace metal;"
        assert code.count(marker) == 1
        helper = MIXED_MATRIX
        if name != "ac_half_storage":
            # Preserve the exact already-screened wrapper for these variants.
            helper = helper.replace("template <typename MatrixA, typename MatrixB>\n", "")
            helper = helper.replace("MatrixA a, MatrixB b,", "simdgroup_float8x8 a, simdgroup_float8x8 b,")
        code = code.replace(marker, marker + "\n" + helper)
    if name == "ac_half_storage":
        start = code.index("inline void GatherAcStrategyPixels(")
        end = code.index("inline void AcStrategyInverseSquareDct(")
        part = code[start:end]
        # The gathered pixels are transient until forward-transform completion;
        # residual computation overwrites the same backing scratch afterward.
        # Preserve its FP32 capacity for that later consumer.
        assert part.count("    pixels[index] = valid") == 1
        part = part.replace("    pixels[index] = valid", "    reinterpret_cast<threadgroup half*>(pixels)[index] = valid")
        part = part.replace("simdgroup_float8x8 a;", "simdgroup_half8x8 a;")
        part = part.replace("        pixels,\n", "        reinterpret_cast<threadgroup half*>(pixels),\n")
        code = code[:start] + part + code[end:]
    return code

def variants(selected):
    baseline = OUT / "baseline"
    for name in selected or VARIANTS:
        shader, flags = VARIANTS[name]
        folder = OUT / "variants" / name
        expected_code = transformed(name, (OUT / "src/src/gpu/metal/kernels" / f"{shader}.metal").read_text())
        if (folder / "identity.json").exists():
            previous = json.loads((folder / "identity.json").read_text())
            if (previous["flags"] != flags or previous["library"] != sha(folder / "metal/gjxl.metallib")
                    or (folder / "kernels" / f"{shader}.metal").read_text() != expected_code):
                raise RuntimeError("Variant changed; assign a new variant name: " + name)
            continue
        kernels = folder / "kernels"
        kernels.mkdir(parents=True, exist_ok=True)
        (folder / "metal").mkdir(exist_ok=True)
        for path in (OUT / "src/src/gpu/metal/kernels").iterdir():
            shutil.copy2(path, kernels / path.name)
        original = (kernels / f"{shader}.metal").read_text()
        code = expected_code
        (kernels / f"{shader}.metal").write_text(code)
        (folder / "shader.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(True), code.splitlines(True), fromfile=f"a/{shader}.metal", tofile=f"b/{shader}.metal")))
        for path in (baseline / "metal").glob("*.ir"):
            shutil.copy2(path, folder / "metal" / path.name)
        revision = sha(kernels / f"{shader}.metal")[:12]
        run(name + "-compile-" + revision, ["xcrun", "-sdk", "macosx", "metal", "-c", *flags,
            kernels / f"{shader}.metal", "-o", folder / "metal" / f"{shader}.ir"])
        inputs = sorted((folder / "metal").glob("*.ir"))
        run(name + "-link", ["xcrun", "-sdk", "macosx", "metallib", *inputs,
                             "-o", folder / "metal/gjxl.metallib"])
        for binary in ("gjxl_encoding_benchmark", "gjxl_metal_precision_probe"):
            destination = folder / binary
            if not destination.exists():
                destination.symlink_to(baseline / binary)
        save(folder / "identity.json", dict(name=name, shader=shader, flags=flags,
            sources={str(p.relative_to(folder)): sha(p) for p in kernels.iterdir()},
            library=sha(folder / "metal/gjxl.metallib"),
            compiler=subprocess.check_output(["xcrun", "metal", "--version"], text=True).strip()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "variants", "metric-probe"))
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "build": build()
    elif args.action == "metric-probe": metric_probe()
    else: variants(args.variants)
