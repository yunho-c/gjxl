#!/usr/bin/env python3
"""Run unchanged contract assertions against an explicitly selected metallib."""
import argparse
import json
import subprocess
from precision import OUT, run, save, sha

TESTS = ("butteraugli_device_operation", "metal_aq_evaluation", "adaptive_quantization_gpu_policy")

def main(variants):
    folder = OUT / "contracts"
    folder.mkdir(exist_ok=True)
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    libraries = ["codestream", "metal", "pfm_io", "gpu_butteraugli", "gpu_ops", "gpu", "codec"]
    probes = {}
    for test in TESTS:
        source = OUT / "src/tests" / (test + "_test.cpp")
        wrapper = folder / (test + ".cpp")
        wrapper.write_text('#include <cstdlib>\n#include <iomanip>\n'
            '#define GJXL_METALLIB_PATH std::getenv("GJXL_PRECISION_METALLIB")\n'
            '#define main PrecisionContractMain\n#include "' + str(source) + '"\n'
            '#undef main\nint main() { std::cerr << std::setprecision(17); return PrecisionContractMain(); }\n')
        binary = folder / test
        run("contract-build-" + test + "-" + sha(wrapper)[:12],
            ["/usr/bin/c++", "-std=c++20", "-stdlib=libc++", "-O3", "-DNDEBUG", "-isysroot", sdk,
             "-I", OUT / "src/src", "-I", OUT / "src/include", "-I", OUT / "src/tests",
             wrapper, "-o", binary, *[OUT / "baseline" / ("libgjxl_" + n + ".a") for n in libraries],
             "-framework", "Metal", "-framework", "Foundation", "-framework", "CoreGraphics"], {"SDKROOT": sdk})
        probes[test] = binary
    for variant in variants:
        library = (OUT / "baseline" if variant == "baseline" else OUT / "variants" / variant) / "metal/gjxl.metallib"
        rows = []
        for test, binary in probes.items():
            result = run("contract-" + variant + "-" + test, [binary],
                         {"GJXL_PRECISION_METALLIB": str(library)}, accepted=(0, 1))
            rows.append(dict(test=test, passed=result["exit"]==0, command=result,
                             binary_sha256=sha(binary), library_sha256=sha(library)))
        save(folder / (variant + ".json"), dict(variant=variant, rows=rows,
             passed=all(row["passed"] for row in rows)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+")
    main(parser.parse_args().variants)
