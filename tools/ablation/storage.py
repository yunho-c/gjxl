"""Bounded decoded scratch and content-addressed evidence for corpus runs.

All work here is outside the timed encoder call. Every measured codestream
remains accessible at its original job path; identical files share an inode.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def pfm_view(path):
    with Path(path).open("rb") as stream:
        def line():
            value = stream.readline().strip()
            while value.startswith(b"#"):
                value = stream.readline().strip()
            return value
        if line() != b"PF":
            raise ValueError("Expected RGB PFM")
        width, height = map(int, line().split())
        scale = float(line())
        offset = stream.tell()
    if width <= 0 or height <= 0 or not np.isfinite(scale) or scale == 0:
        raise ValueError("Invalid PFM header")
    count = width * height * 3
    if Path(path).stat().st_size != offset + count * 4:
        raise ValueError("Invalid PFM payload size")
    values = np.memmap(path, dtype="<f4" if scale < 0 else ">f4",
                       mode="r", offset=offset, shape=(count,))
    return (width, height), abs(scale), values


def pixel_delta(left, right):
    le, ls, lv = pfm_view(left)
    re, rs, rv = pfm_view(right)
    if le != re:
        raise ValueError("Decoded extent mismatch")
    total, maximum = 0., 0.
    for start in range(0, len(lv), 1 << 20):
        a = np.array(lv[start:start + (1 << 20)], dtype=np.float64, subok=False) * ls
        b = np.array(rv[start:start + (1 << 20)], dtype=np.float64, subok=False) * rs
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Non-finite decoded pixels")
        difference = np.abs(a - b)
        maximum = max(maximum, float(np.max(difference)))
        total += float(np.sum(difference * difference))
    return {"rmse": float(np.sqrt(total / len(lv))), "max_abs": maximum}


class CompactStore:
    def __init__(self, root, decoder, metric, command):
        self.root = Path(root)
        self.decoder, self.metric, self.command = decoder, metric, command
        self.tools = {"decoder": sha(decoder), "metric": sha(metric)}
        for name in ("objects", "validation", "pair-validation", "scratch"):
            (self.root / name).mkdir(exist_ok=True)

    def decode(self, encoded, decoded):
        self.command([self.decoder, encoded, decoded, "--num_threads=2",
                      "--color_space=RGB_D65_SRG_Rel_Lin"], timeout=600)

    def validate(self, image, input_sha, directory):
        encoded = directory / "output.jxl"
        digest = sha(encoded)
        object_path = self.root / "objects" / (digest + ".jxl")
        if object_path.exists():
            if sha(object_path) != digest:
                raise ValueError("Stored codestream object changed")
            link = directory / "output.link"
            if link.exists():
                link.unlink()
            os.link(object_path, link)
            link.replace(encoded)
        else:
            os.link(encoded, object_path)
        cache = self.root / "validation" / (input_sha + "-" + digest + ".json")
        identity = {"input_sha256": input_sha, "codestream_sha256": digest,
                    "tools": self.tools}
        if cache.exists():
            result = json.loads(cache.read_text())
            if result["identity"] != identity:
                raise ValueError("Validation cache identity changed")
        else:
            with tempfile.TemporaryDirectory(prefix="decode-", dir=self.root / "scratch") as tmp:
                decoded = Path(tmp) / "decoded.pfm"
                self.decode(encoded, decoded)
                distortion = pixel_delta(image, decoded)
                metric_text = self.command([self.metric, image, decoded], timeout=600)
                quality = float(metric_text)
                if not np.isfinite(quality):
                    raise ValueError("Non-finite quality score")
                result = {"identity": identity, "decoded_sha256": sha(decoded),
                          "input_pixel_error": distortion, "quality": quality,
                          "metric_text": metric_text}
            save(cache, result)
        (directory / "metric.txt").write_text(result["metric_text"] + "\n")
        return result | {"validation_cache": str(cache.relative_to(self.root)),
                         "validation_sha256": sha(cache)}

    def pair_delta(self, a, b):
        if a["decoded_sha256"] == b["decoded_sha256"]:
            return {"rmse": 0., "max_abs": 0.}
        hashes = sorted((a["artifacts"]["output.jxl"], b["artifacts"]["output.jxl"]))
        cache = self.root / "pair-validation" / ("-".join(hashes) + ".json")
        identity = {"codestreams": hashes, "decoder_sha256": self.tools["decoder"]}
        if cache.exists():
            result = json.loads(cache.read_text())
            if result["identity"] != identity:
                raise ValueError("Pair cache identity changed")
            return result["delta"]
        with tempfile.TemporaryDirectory(prefix="pair-", dir=self.root / "scratch") as tmp:
            paths = []
            for index, record in enumerate((a, b)):
                decoded = Path(tmp) / f"{index}.pfm"
                self.decode(self.root / record["id"] / "output.jxl", decoded)
                if sha(decoded) != record["decoded_sha256"]:
                    raise ValueError("Decoder did not reproduce validated pixels")
                paths.append(decoded)
            delta = pixel_delta(*paths)
        save(cache, {"identity": identity, "delta": delta})
        return delta
