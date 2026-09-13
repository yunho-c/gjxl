#!/usr/bin/env python3
"""Select synthetic or photographic inputs for the image batch benchmark."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools/benchmark_corpora/photo-large.json"
STUDY_CORPUS = ROOT.parent / "libjxl-runtime-study-2026-09-03/corpus"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Corpus manifest path must be relative: {relative}")
    return root / path


def verify(path, digest):
    if not path.is_file():
        raise ValueError(
            f"Missing corpus file: {path}. Use --photo-root CORPUS or --prepare-photos."
        )
    if sha256(path) != digest:
        raise ValueError(
            f"Corpus SHA-256 mismatch: {path}; existing files are never replaced."
        )


def verified_images(manifest, root):
    images = []
    for photo in manifest["photos"]:
        for entry in photo["images"]:
            path = corpus_path(root, entry["file"])
            verify(path, entry["sha256"])
            if path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Corpus file size mismatch: {path}")
            images.append({**entry, "path": str(path.resolve())})
    return sorted(images, key=lambda image: image["id"])


def prepare_photos(manifest, root, magick):
    """Explicit preparation only; benchmark execution never downloads or resizes."""
    for photo in manifest["photos"]:
        missing = [
            entry
            for entry in photo["images"]
            if not corpus_path(root, entry["file"]).exists()
        ]
        for entry in photo["images"]:
            path = corpus_path(root, entry["file"])
            if path.exists():
                verify(path, entry["sha256"])
        if not missing:
            continue
        executable = shutil.which(magick)
        if executable is None:
            raise ValueError(
                "ImageMagick is required to prepare missing photographic inputs"
            )
        source = corpus_path(root, photo["source_file"])
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            with tempfile.TemporaryDirectory(
                prefix="photo-download-", dir=source.parent
            ) as temporary:
                downloaded = Path(temporary) / "source.jpg"
                print(f"Downloading {photo['id']}", file=sys.stderr, flush=True)
                with (
                    urllib.request.urlopen(
                        photo["download_url"], timeout=60
                    ) as response,
                    downloaded.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out)
                verify(downloaded, photo["source_sha256"])
                # An exclusive hard link publishes only a complete, verified file.
                os.link(downloaded, source)
        verify(source, photo["source_sha256"])
        for entry in missing:
            destination = corpus_path(root, entry["file"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="photo-resize-", dir=destination.parent
            ) as temporary:
                candidate = Path(temporary) / "image.pfm"
                replacements = {
                    "SOURCE": str(source),
                    "DESTINATION.pfm": str(candidate),
                    "WIDTHxHEIGHT!": f"{entry['width']}x{entry['height']}!",
                }
                argv = [
                    executable,
                    *(
                        replacements.get(arg, arg)
                        for arg in manifest["preparation"]["magick_arguments"]
                    ),
                ]
                print(f"Preparing {entry['id']}", file=sys.stderr, flush=True)
                subprocess.run(argv, check=True)
                if (
                    sha256(candidate) != entry["sha256"]
                    or candidate.stat().st_size != entry["bytes"]
                ):
                    raise ValueError(
                        f"Prepared pixels differ from the pinned corpus: {entry['id']}. "
                        f"Original preparation used {manifest['prepared_with']}"
                    )
                os.link(candidate, destination)
    return verified_images(manifest, root)


def batch_sizes(value, photographic=False):
    if value == "auto":
        # The 12MP photographs round just below 12,000,000 pixels. Select one
        # explicitly so the native per-area policy cannot admit 2/4/8 copies.
        return "1" if photographic else "auto"
    parts = value.split(",")
    if not all(part.isascii() and part.isdigit() and int(part) > 0 for part in parts):
        raise ValueError(
            "Batch sizes must be auto or comma-separated positive integers"
        )
    return value


def main(argv=None, *, manifest_path=MANIFEST):
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog="Other options are forwarded to gjxl_image_batch_benchmark. "
        "photo-large uses nine pinned PFMs; synthetic-large uses generated stress fixtures.",
    )
    parser.add_argument(
        "--benchmark", default=str(ROOT / "build/release/gjxl_image_batch_benchmark")
    )
    parser.add_argument("--workload", default="all")
    parser.add_argument("--batch-sizes", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw-samples", type=Path)
    parser.add_argument(
        "--photo-root",
        type=Path,
        default=os.environ.get(
            "GJXL_PHOTO_CORPUS",
            str(
                STUDY_CORPUS
                if STUDY_CORPUS.is_dir()
                else ROOT / "build/benchmark-corpus/photo-large"
            ),
        ),
        help="Corpus root containing pfm/unsplash (default: existing sibling study, otherwise build cache)",
    )
    parser.add_argument(
        "--prepare-photos",
        action="store_true",
        help="Download/prepare missing pinned photos, then exit without encoding",
    )
    parser.add_argument("--magick", default=os.environ.get("GJXL_MAGICK", "magick"))
    args, forwarded = parser.parse_known_args(argv)
    if args.dry_run and args.raw_samples:
        raise ValueError("--dry-run cannot write --raw-samples")
    if args.prepare_photos and (args.dry_run or args.raw_samples or forwarded):
        raise ValueError(
            "--prepare-photos cannot be combined with benchmark or dry-run arguments"
        )
    photographic = args.workload == "photo-large"
    if photographic and any(
        arg.split("=", 1)[0] in {"--input", "--export-inputs"} for arg in forwarded
    ):
        raise ValueError(
            "photo-large selects only its manifest inputs; --input and --export-inputs are not supported"
        )
    sizes = batch_sizes(args.batch_sizes, photographic)
    root = args.photo_root.expanduser().resolve()
    manifest = None
    images = []
    if photographic or args.prepare_photos:
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest["schema_version"] != 1 or manifest["corpus_id"] != "photo-large":
            raise ValueError("Unsupported photo corpus manifest")
        if args.prepare_photos:
            images = prepare_photos(manifest, root, args.magick)
            print(
                f"Verified {len(images)} photographic inputs in {root}; no encoding performed."
            )
            return 0
        images = verified_images(manifest, root)
        if args.dry_run:
            writer = csv.writer(sys.stdout, lineterminator="\n")
            writer.writerow(
                [
                    "corpus",
                    "workload",
                    "width",
                    "height",
                    "batch_sizes",
                    "source",
                    "sha256",
                ]
            )
            for image in images:
                writer.writerow(
                    [
                        "photo-large",
                        image["id"],
                        image["width"],
                        image["height"],
                        sizes,
                        image["path"],
                        image["sha256"],
                    ]
                )
            return 0
    benchmark = shutil.which(args.benchmark)
    if benchmark is None:
        raise ValueError(f"Benchmark executable not found: {args.benchmark}")
    command = [benchmark, "--batch-sizes", sizes]
    if photographic:
        for image in images:
            command.extend(["--input", image["path"]])
    else:
        command.extend(["--workload", args.workload])
        if args.dry_run:
            command.append("--dry-run")
    command.extend(forwarded)
    if args.raw_samples:
        command.extend(["--raw-samples", str(args.raw_samples)])
        if args.raw_samples.exists():
            raise ValueError(f"Raw sample output already exists: {args.raw_samples}")
        if photographic:
            sidecar = args.raw_samples.with_name(args.raw_samples.name + ".inputs.json")
            with sidecar.open("x") as stream:
                json.dump(
                    {
                        "schema_version": 1,
                        "corpus": "photo-large",
                        "manifest": str(Path(manifest_path).resolve()),
                        "manifest_sha256": sha256(manifest_path),
                        "inputs": images,
                        "benchmark_sha256": sha256(benchmark),
                        "cwd": str(Path.cwd()),
                        "command": command,
                    },
                    stream,
                    indent=2,
                )
                stream.write("\n")
    if photographic:
        print(
            f"photo-large: verified {len(images)} inputs; batch sizes {sizes}",
            file=sys.stderr,
            flush=True,
        )
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Benchmark input error: {error}", file=sys.stderr)
        raise SystemExit(1)
