#!/usr/bin/env python3
"""Decode each distinct qualified bitstream using the independent libjxl CLI."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DECODER = Path('/opt/homebrew/bin/djxl')


def main():
    records = json.loads((ROOT / 'final-profile/results.json').read_text())
    unique = {(r['image'], r['effort']): r for r in records if r['round'] == 0 and r['mode'] == 'base'}
    assert len(unique) == 18
    out = ROOT / 'decode'
    out.mkdir(exist_ok=True)
    result = dict(decoder=str(DECODER), decoder_sha256=hashlib.sha256(DECODER.read_bytes()).hexdigest(),
                  version=subprocess.check_output([str(DECODER), '--version'], stderr=subprocess.STDOUT, text=True), rows=[])
    for (name, effort), r in unique.items():
        folder = ROOT / 'final-profile' / f'r0-{name}-e{effort}-base'
        info = json.loads((folder / 'paired.json').read_text())
        encoded = folder / 'reference.jxl'
        command = [str(DECODER), str(encoded), '-', '--output_format', 'pfm', '--quiet']
        with (out / (name.replace('/', '_') + f'-e{effort}.log')).open('w') as log:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
            digest, count, prefix = hashlib.sha256(), 0, b''
            while chunk := proc.stdout.read(1024 * 1024):
                digest.update(chunk)
                count += len(chunk)
                if not prefix:
                    prefix = chunk[:512]
            assert proc.wait() == 0, (name, effort, 'decode failed')
        lines = prefix.split(b'\n', 3)
        assert lines[0] == b'PF', lines[:3]
        width, height = map(int, lines[1].split())
        assert (width, height) == (info['source_width'], info['source_height'])
        header_bytes = sum(len(x) + 1 for x in lines[:3])
        assert count == header_bytes + width * height * 12
        result['rows'].append(dict(image=name, effort=effort, argv=command, output_sha256=r['output_sha256'],
            decoded_pfm_sha256=digest.hexdigest(), decoded_bytes=count, width=width, height=height))
        (out / 'validation.json').write_text(json.dumps(result, indent=2))
        print(name, effort, 'decoded', flush=True)


if __name__ == '__main__':
    main()
