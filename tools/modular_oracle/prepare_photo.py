"""Reproduce photo.ppm from the pinned CC0 wesaturate fixture (requires Pillow)."""
import hashlib
import sys
from pathlib import Path
from PIL import Image

source, output = map(Path, sys.argv[1:])
assert hashlib.sha256(source.read_bytes()).hexdigest() == "fa85b95a14533303b64d9015537a6f21e50094dcc42ca9c48f93ed242a9781f6"
image = Image.open(source).convert("RGB")
assert image.size == (500, 500)
pixels = bytes(v for y in range(0, 500, 4) for x in range(0, 500, 4) for v in image.getpixel((x, y)))
output.write_bytes(b"P6\n125 125\n255\n" + pixels)
