"""Build the standalone decoder timing helper against an explicit pinned build."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, build, output = (p.resolve() for p in (args.source, args.build, args.output))
    revision = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if revision != "e8ff09762481785938d8e4e01333ed3917571161":
        raise SystemExit("Decoder source is not the pinned revision")
    subprocess.run(["git", "-C", source, "diff", "--quiet", "HEAD", "--"], check=True)
    output.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).with_name("decode_benchmark.cpp").resolve()
    binary = output / "decode_benchmark"
    compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    command = [compiler, "-isysroot", sdk, "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra",
               "-Werror", "-Wno-deprecated-declarations", "-I" + str(source / "lib/include"),
               "-I" + str(build / "lib/include"), str(helper), str(build / "lib/libjxl.dylib"),
               "-Wl,-rpath," + str(build / "lib"), "-o", str(binary)]
    subprocess.run(command, check=True)
    libraries = sorted({p.resolve() for folder in (build / "lib", build / "third_party/brotli")
                        for p in folder.glob("*.dylib")})
    manifest = {"command": command, "revision": revision, "binary": str(binary),
                "binary_sha256": sha(binary), "source_sha256": sha(helper),
                "builder_sha256": sha(__file__), "cmake_cache_sha256": sha(build / "CMakeCache.txt"),
                "compiler": subprocess.check_output([compiler, "--version"], text=True),
                "libraries": {str(p): sha(p) for p in libraries},
                "linked_libraries": subprocess.check_output(["otool", "-L", binary], text=True)}
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()
