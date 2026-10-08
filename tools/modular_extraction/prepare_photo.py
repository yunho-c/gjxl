"""Make the tiled, linear-float photographic fixture (requires Pillow)."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = Image.open(args.source).convert("RGB")
    width, height = source.size
    lookup = [struct.pack("<f", v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
              for v in (i / 255 for i in range(256))]
    pixels = source.tobytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(f"PF\n{width * 5} {height}\n-1.0\n".encode("ascii"))
        for y in range(height - 1, -1, -1):
            row = b"".join(lookup[v] for v in pixels[y * width * 3:(y + 1) * width * 3])
            output.write(row * 5)
    record = {"source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "preparation": "five horizontal copies; exact sRGB transfer to float32 linear RGB",
              "extent": [width * 5, height], "pfm_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
