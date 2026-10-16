#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Check retained ANS models against the pre-storage-refactor byte oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--golden', type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.golden.read_text())['cases']
    with tempfile.TemporaryDirectory(prefix='gjxl-ans-search-') as temporary:
        out = Path(temporary) / 'probe'
        subprocess.run([str(args.binary.resolve()), str(out)], check=True)
        actual = [json.loads(line) for line in (out / 'results.jsonl').read_text().splitlines()]
        if len(actual) != len(expected):
            raise RuntimeError('ANS fixture coverage changed')
        for before, after in zip(expected, actual):
            for key in ['case', 'bits', 'configs', 'histograms']:
                if before[key] != after[key]:
                    raise RuntimeError(f'ANS {before["case"]}: {key} differs')
            checksum = hashlib.sha256((out / (after['case'] + '.bin')).read_bytes()).hexdigest()
            if checksum != before['sha256']:
                raise RuntimeError(f'ANS {before["case"]}: retained model/payload bytes differ')
            if after['peak_backing'] > after['plan']:
                raise RuntimeError('ANS storage exceeded admission')
        print(f'{len(actual)} frozen ANS model/payloads and candidate searches match; '
              f'{sum(row["allocations"] for row in actual)} managed allocations')


if __name__ == '__main__':
    main()
