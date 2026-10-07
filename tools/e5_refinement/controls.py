#!/usr/bin/env python3
"""Byte controls for unchanged efforts/overrides and e5 scored-output parity."""
import argparse
from pathlib import Path

from qualify import Study, check, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    study = Study(root)
    images = {i['image_id']: i for i in study.m['images']}
    name = 'kodak/01'
    tests = [(f'e{e}', name, ['--effort', str(e), '--distance', '1.9'])
             for e in (1, 2, 3, 4, 6, 7, 8, 9, 10)]
    tests += [
        # The CLI makes --effort and --high-density mutually exclusive. The
        # policy unit test separately covers high density at every effort.
        ('high-density', name, ['--distance', '1.9', '--high-density']),
        ('maximum-error', name, ['--effort', '5', '--maximum-error', '.2', '.2', '.2']),
        ('maximum-throughput', name, ['--effort', '5', '--distance', '1.9', '--gpu-aq', 'maximum-throughput']),
        ('cpu-e6', name, ['--effort', '6', '--distance', '1.9', '--backend', 'cpu']),
        ('exact-e6', name, ['--effort', '6', '--distance', '1.9', '--gpu-aq', 'exact-coefficients']),
    ]
    tests += [(f'photo-{index}-e{e}', n, ['--effort', str(e), '--distance', '1.9'])
              for index, n in enumerate([
                  'clic2024_test/097cb426910ba8ce2525dd8bb7fb1777',
                  'unsplash/campus_interior/12mp']) for e in (4, 6)]
    results = []
    def encode(label, image, arm, flags):
        folder = root / 'controls' / label / arm
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / 'out.jxl'
        base = ['--backend', 'metal', '--gpu-aq', 'fully-resident']
        if '--backend' in flags:
            base = base[2:]
        if '--gpu-aq' in flags:
            base = base[:2] if '--backend' not in flags else []
        log = study.command([str(Path(study.m[arm]).with_name('gjxl_encode')),
                             *base, *flags, images[image]['pfm_path'], output])
        (folder / 'stdout.txt').write_text(log)
        return sha(output)
    for label, image, flags in tests:
        a = encode(label, image, 'baseline', flags)
        b = encode(label, image, 'candidate', flags)
        check(a == b, f'Unchanged policy changed bytes: {label}')
        results.append(dict(label=label, image_id=image, flags=flags, sha256=a, identical=True))
        save(root / 'controls.json', dict(complete=False, results=results))
    for mode in ('fully-resident', 'exact-coefficients', 'cpu'):
        flags = ['--effort', '5', '--distance', '1.9']
        flags += ['--backend', 'cpu'] if mode == 'cpu' else ['--gpu-aq', mode]
        a = encode(f'e5-{mode}-unscored', name, 'candidate', flags)
        b = encode(f'e5-{mode}-scored', name, 'candidate', flags + ['--collect-final-score'])
        check(a == b, f'Final diagnostics changed bytes: {mode}')
        results.append(dict(label=f'e5-{mode}-scored', identical=True, sha256=a))
    for mode in ('metal', 'cpu'):
        flags = ['--effort', '5', '--backend', mode, '--max-attempts', '8']
        a = encode(f'e5-{mode}-target-bytes', name, 'candidate',
                   flags + ['--target-bytes', '30000'])
        b = encode(f'e5-{mode}-target-bpp', name, 'candidate',
                   flags + ['--target-bpp', str(30000 * 8 / images[name]['pixels'])])
        c = encode(f'e5-{mode}-target-scored', name, 'candidate',
                   flags + ['--target-bytes', '30000', '--collect-final-score'])
        check(a == b == c, f'Target retry/BPP/diagnostic parity failed: {mode}')
        results.append(dict(label=f'e5-{mode}-target-retries', identical=True, sha256=a))
    save(root / 'controls.json', dict(complete=True, results=results,
                                     runner_sha256=sha(__file__), checks=len(results)))
    print(f'{len(results)} byte controls passed')


if __name__ == '__main__':
    main()
