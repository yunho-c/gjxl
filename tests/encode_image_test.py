#!/usr/bin/env python3
"""Tests the optional ImageMagick input wrapper around gjxl_encode."""

from __future__ import annotations

import argparse
import binascii
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(
    path: Path,
    width: int,
    height: int,
    color_type: int,
    channels: int,
    pixels: bytes,
) -> None:
    if len(pixels) != width * height * channels:
        raise ValueError("PNG fixture has the wrong pixel count")
    stride = width * channels
    rows = b"".join(
        b"\0" + pixels[y * stride : (y + 1) * stride]
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )


def read_pfm(path: Path) -> tuple[int, int, list[tuple[float, float, float]]]:
    with path.open("rb") as stream:
        if stream.readline() != b"PF\n":
            raise ValueError("expected an RGB PFM")
        width, height = (int(value) for value in stream.readline().split())
        scale = float(stream.readline())
        payload = stream.read()
    endian = "<" if scale < 0.0 else ">"
    values = struct.unpack(f"{endian}{width * height * 3}f", payload)
    file_rows = [
        [tuple(values[3 * (y * width + x) : 3 * (y * width + x + 1)])
         for x in range(width)]
        for y in range(height)
    ]
    pixels = [pixel for row in reversed(file_rows) for pixel in row]
    return width, height, pixels


def srgb_to_linear(value: int) -> float:
    encoded = value / 255.0
    if encoded <= 0.04045:
        return encoded / 12.92
    return ((encoded + 0.055) / 1.055) ** 2.4


def add_exif_orientation(path: Path, orientation: int) -> None:
    jpeg = path.read_bytes()
    if not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("EXIF fixture is not a JPEG")
    tiff = (
        b"MM\x00*\x00\x00\x00\x08"
        + b"\x00\x01"
        + struct.pack(">HHI", 0x0112, 3, 1)
        + struct.pack(">H", orientation)
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"
    )
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    path.write_bytes(jpeg[:2] + app1 + jpeg[2:])


class EncodeImageTest(unittest.TestCase):
    wrapper: Path
    encoder: Path
    magick: Path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gjxl-wrapper-test-")
        self.directory = Path(self.temporary.name)
        self.copy_encoder = self.directory / "copy_encoder.py"
        self.copy_encoder.write_text(
            "from pathlib import Path\n"
            "import shutil\n"
            "import sys\n"
            "shutil.copyfile(Path(sys.argv[-2]), Path(sys.argv[-1]))\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_wrapper(
        self,
        source: Path,
        destination: Path,
        *,
        magick: str | None = None,
        real_encoder: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        encoder = str(self.encoder if real_encoder else Path(sys.executable))
        command = [
            sys.executable,
            str(self.wrapper),
            "--encoder",
            encoder,
            "--magick",
            magick or str(self.magick),
            str(source),
            str(destination),
            "--",
        ]
        if real_encoder:
            command.extend(["--distance", "1.0", "--backend", "cpu"])
        else:
            command.append(str(self.copy_encoder))
        return subprocess.run(command, check=False, capture_output=True, text=True)

    def test_rgb_png_becomes_linear_rgb_pfm(self) -> None:
        source = self.directory / "rgb.png"
        destination = self.directory / "rgb.pfm"
        write_png(source, 1, 1, 2, 3, bytes((128, 64, 32)))

        result = self.run_wrapper(source, destination)

        self.assertEqual(result.returncode, 0, result.stderr)
        width, height, pixels = read_pfm(destination)
        self.assertEqual((width, height), (1, 1))
        expected = tuple(srgb_to_linear(value) for value in (128, 64, 32))
        for actual, target in zip(pixels[0], expected):
            self.assertAlmostEqual(actual, target, delta=3.0e-5)

    def test_grayscale_png_is_forced_to_three_channels(self) -> None:
        source = self.directory / "gray.png"
        destination = self.directory / "gray.pfm"
        write_png(source, 2, 1, 0, 1, bytes((0, 128)))

        result = self.run_wrapper(source, destination)

        self.assertEqual(result.returncode, 0, result.stderr)
        width, height, pixels = read_pfm(destination)
        self.assertEqual((width, height), (2, 1))
        self.assertEqual(pixels[0], (0.0, 0.0, 0.0))
        for channel in pixels[1]:
            self.assertAlmostEqual(channel, srgb_to_linear(128), delta=3.0e-5)

    def test_alpha_is_composited_over_linear_white(self) -> None:
        source = self.directory / "alpha.png"
        destination = self.directory / "alpha.pfm"
        write_png(source, 1, 1, 6, 4, bytes((255, 0, 0, 128)))

        result = self.run_wrapper(source, destination)

        self.assertEqual(result.returncode, 0, result.stderr)
        _, _, pixels = read_pfm(destination)
        self.assertAlmostEqual(pixels[0][0], 1.0, delta=3.0e-5)
        expected_uncovered = 127.0 / 255.0
        self.assertAlmostEqual(pixels[0][1], expected_uncovered, delta=3.0e-5)
        self.assertAlmostEqual(pixels[0][2], expected_uncovered, delta=3.0e-5)

    def test_exif_orientation_is_applied(self) -> None:
        source = self.directory / "oriented.jpg"
        destination = self.directory / "oriented.pfm"
        subprocess.run(
            [str(self.magick), "-size", "2x1", "xc:red", str(source)],
            check=True,
            capture_output=True,
        )
        add_exif_orientation(source, 6)

        result = self.run_wrapper(source, destination)

        self.assertEqual(result.returncode, 0, result.stderr)
        width, height, _ = read_pfm(destination)
        self.assertEqual((width, height), (1, 2))

    def test_multiframe_input_is_rejected_atomically(self) -> None:
        source = self.directory / "animated.gif"
        destination = self.directory / "sentinel.jxl"
        subprocess.run(
            [
                str(self.magick),
                "-size",
                "2x2",
                "xc:red",
                "-size",
                "2x2",
                "xc:blue",
                "-delay",
                "10",
                "-loop",
                "0",
                str(source),
            ],
            check=True,
            capture_output=True,
        )
        destination.write_bytes(b"unchanged")

        result = self.run_wrapper(source, destination)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("multi-page input is unsupported", result.stderr)
        self.assertEqual(destination.read_bytes(), b"unchanged")

    def test_pfm_does_not_require_imagemagick(self) -> None:
        source = self.directory / "source.pfm"
        destination = self.directory / "copied.pfm"
        source.write_bytes(b"PF\n1 1\n-1.0\n" + struct.pack("<3f", 0.1, 0.2, 0.3))

        result = self.run_wrapper(
            source,
            destination,
            magick="gjxl-definitely-missing-magick",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_missing_imagemagick_preserves_destination(self) -> None:
        source = self.directory / "source.png"
        destination = self.directory / "sentinel.jxl"
        write_png(source, 1, 1, 2, 3, bytes((1, 2, 3)))
        destination.write_bytes(b"unchanged")

        result = self.run_wrapper(
            source,
            destination,
            magick="gjxl-definitely-missing-magick",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ImageMagick executable was not found", result.stderr)
        self.assertEqual(destination.read_bytes(), b"unchanged")

    def test_png_encodes_with_the_real_cli(self) -> None:
        source = self.directory / "sample.png"
        destination = self.directory / "sample.jxl"
        pixels = bytearray()
        for y in range(13):
            for x in range(17):
                pixels.extend((x * 15, y * 19, (7 * x + 11 * y) % 256))
        write_png(source, 17, 13, 2, 3, bytes(pixels))

        result = self.run_wrapper(source, destination, real_encoder=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(destination.read_bytes().startswith(b"\xff\x0a"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--magick", type=Path, required=True)
    args, unittest_args = parser.parse_known_args()
    EncodeImageTest.wrapper = args.wrapper
    EncodeImageTest.encoder = args.encoder
    EncodeImageTest.magick = args.magick
    unittest.main(argv=[sys.argv[0], *unittest_args])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
