#!/usr/bin/env python3
"""Reconstruct and qualify resident execution using explicit dependencies."""
import argparse
import json
import sys
from pathlib import Path
from session import BASELINE, Session, configure
import gates
import evidence
import protocol


def main(argv=None):
    if sys.flags.optimize:
        raise RuntimeError('Run without Python optimization; protocol assertions are qualification gates')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Dedicated persistent output directory')
    actions = parser.add_subparsers(dest='action', required=True)
    config = actions.add_parser('configure')
    config.add_argument('--repo', type=Path, required=True)
    config.add_argument('--baseline-ref', default=BASELINE)
    candidate = config.add_mutually_exclusive_group()
    candidate.add_argument('--candidate-ref', default='HEAD')
    candidate.add_argument('--historical', action='store_true', help='Compare ec4d4c5 with 07dd92e')
    config.add_argument('--baseline-capability', choices=('legacy', 'current'), default='legacy')
    config.add_argument('--corpus', type=Path, required=True, help='Directory containing canonical PFMs from corpus.json')
    config.add_argument('--decoder', type=Path, required=True)
    config.add_argument('--info', type=Path, required=True)
    config.add_argument('--jobs', type=int, default=8)
    config.add_argument('--profile', choices=('smoke', 'full'), default='smoke')
    for name in ('build', 'identities', 'smoke', 'tests', 'sanitizers', 'parity', 'conformance', 'retained',
                 'performance', 'pressure', 'report', 'audit', 'seal', 'verify', 'status', 'all'):
        actions.add_parser(name)
    args = parser.parse_args(argv)
    if args.action == 'configure':
        if args.jobs < 1:
            parser.error('--jobs must be positive')
        return configure(args)
    session = Session(args.out)
    protocol.initialize(session)
    action = args.action
    if action == 'status':
        for name in ('build', 'tests', 'sanitizers', 'parity', 'conformance', 'retained', 'performance', 'pressure', 'validation'):
            path = session.out / (name + '.json')
            if path.exists():
                data = json.loads(path.read_text())
                print(name, data.get('status'), data.get('profile', session.config['profile']))
        return
    if action == 'all':
        for step in ('build', 'smoke', 'tests', 'sanitizers', 'parity', 'conformance', 'retained', 'performance', 'pressure', 'seal', 'verify'):
            dispatch(session, step)
    else:
        dispatch(session, action)


def dispatch(session, action):
    if action in ('build', 'identities'):
        return getattr(session, action)()
    if action in ('tests', 'sanitizers'):
        return getattr(gates, action)(session)
    if action in ('audit', 'seal', 'verify'):
        return getattr(evidence, action)(session)
    session.identities()
    return getattr(protocol, action)()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, AssertionError, OSError, ValueError) as error:
        print('Qualification failed: ' + str(error), file=sys.stderr)
        sys.exit(1)
