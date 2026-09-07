"""Unit tests for the qualification controller; no Metal or corpus needed."""
import argparse
import contextlib
import io
import json
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import evidence
import gates
import protocol
import run
from session import Session, check, read, save, sha, validate_corpus


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qualification with spaces ')
        self.root = Path(self.temp.name)
        self.config = {'repo': str(self.root), 'revisions': {'integrated': 'a', 'candidate': 'b'},
                      'sdk': str(self.root/'selected SDK'),
                      'capabilities': {'integrated': 'legacy', 'candidate': 'current'},
                      'corpus': str(self.root), 'decoder': str(self.root/'djxl'), 'info': str(self.root/'jxlinfo'),
                      'profile': 'smoke', 'inputs': {r['canonical_path']:r['canonical_sha256'] for r in read(Path(__file__).with_name('corpus.json'))['inputs']}}
        save(self.root / 'config.json', self.config)
        self.session = Session(self.root)
        protocol.initialize(self.session)

    def tearDown(self):
        self.temp.cleanup()

    def test_profiles_have_exact_membership(self):
        self.assertEqual(len(protocol.cases()), 5)
        self.assertEqual(protocol.pairs(), 2)
        self.assertEqual(sum(len(limits)*2 for _,_,limits in protocol.pressure_tasks()), 6)
        self.config['profile'] = 'full'
        self.session.config['profile'] = 'full'
        self.assertEqual(len(protocol.cases()), 27)
        self.assertEqual(protocol.pairs(), 7)
        self.assertEqual(sum(len(limits)*2*3 for _,_,limits in protocol.pressure_tasks()), 42)
        self.assertEqual(len(protocol.parity_tasks()), 56)
        self.assertEqual(len({n for n,_,_ in protocol.parity_tasks()}), 56)

    def test_missing_and_changed_corpus(self):
        manifest = {'inputs': [{'canonical_path':'image with spaces.pfm','canonical_sha256':'wrong'}]}
        with patch('session.read', return_value=manifest):
            with self.assertRaisesRegex(RuntimeError, 'Missing canonical'):
                validate_corpus(self.root)
            image = self.root/'image with spaces.pfm'
            image.write_bytes(b'PF\n1 1\n-1\n')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                validate_corpus(self.root)
            manifest['inputs'][0]['canonical_sha256'] = sha(image)
            self.assertEqual(validate_corpus(self.root), {'image with spaces.pfm':sha(image)})

    def test_commands_preserve_arguments_and_interrupted_logs(self):
        argument = 'a path with spaces and $literal'
        record = self.session.run([sys.executable, '-c',
            'import os,sys; print(sys.argv[1]); print(os.environ["SDKROOT"])', argument], 'command')
        self.assertEqual((self.root/record['log']).read_text().splitlines(), [argument,self.config['sdk']])
        with self.assertRaises(subprocess.TimeoutExpired):
            self.session.run([sys.executable, '-c', 'import time; time.sleep(10)'], 'interrupted', timeout=0.05)
        retry = self.session.run([sys.executable, '-c', 'print("retry")'], 'interrupted')
        self.assertEqual(retry['log'], 'interrupted-attempt1.log')
        self.assertTrue((self.root/'interrupted.log').is_file())
        events = [json.loads(line) for line in (self.root/'execution-events.jsonl').read_text().splitlines()]
        self.assertTrue(any(e['event'] == 'interrupted' for e in events))

    def test_resume_requires_same_driver_configuration_and_log(self):
        (self.root/'integrated-driver').write_text('driver')
        save(self.root/'build.json', {'identity':'fixed'})
        setup = {'type':'setup'}
        raw = [setup, {'type':'sample','index':0}, {'type':'trim'}]
        log = self.root/'sample.log'
        log.write_text('\n'.join(map(json.dumps, raw)))
        record = {'command':[], 'log':log.name, 'log_sha256':sha(log), 'exit':0}
        with patch.object(protocol, 'run', return_value=record) as execute:
            a = protocol.sample_job('sample', 'integrated', ['--synthetic','17x9'], count=1, measured=False)
            b = protocol.sample_job('sample', 'integrated', ['--synthetic','17x9'], count=1, measured=False)
            self.assertEqual(a,b)
            self.assertEqual(execute.call_count,1)
            with self.assertRaises(AssertionError):
                protocol.sample_job('sample','integrated', ['--synthetic','18x9'], count=1, measured=False)
            log.write_text('tampered')
            with self.assertRaises(AssertionError):
                protocol.sample_job('sample','integrated', ['--synthetic','17x9'], count=1, measured=False)

    def test_verify_does_not_rewrite_and_rejects_tampering(self):
        payload = self.root/'result.json'
        save(payload, {'result':1})
        save(self.root/'validation.json', {'profile':'smoke','full_qualification':False,'hashes':{'result.json':sha(payload)}})
        before = {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.root.iterdir()}
        with patch('evidence.audit'):
            evidence.verify(self.session)
            self.assertEqual(before, {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.root.iterdir()})
            payload.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'artifact changed'):
                evidence.verify(self.session)

    def test_partial_profile_cannot_be_promoted(self):
        save(self.root/'validation.json', {'profile':'full','full_qualification':True,'hashes':{}})
        with self.assertRaisesRegex(RuntimeError,'scope changed'):
            evidence.verify(self.session)
        with patch.object(self.session,'identities'):
            for name in ('tests','sanitizers','parity','conformance','retained','performance','pressure','performance-report'):
                save(self.root/(name+'.json'), {'status':'running'})
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                evidence.audit(self.session)

    def test_quiet_check_handles_spaces_in_executable_paths(self):
        state = 'PID %CPU ELAPSED COMMAND\n123 0.0 00:01 /path with spaces/candidate-driver --count 9'
        with patch('session.output', side_effect=[state, '123 /path with spaces/candidate-driver']):
            with self.assertRaisesRegex(RuntimeError, 'quiet encoder/build'):
                self.session.quiet()
        with patch('session.output', side_effect=[state.replace('candidate-driver', 'unrelated'), '123 /path with spaces/unrelated']):
            self.session.quiet()

    def test_configuration_must_supply_dependencies(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.main(['--out',str(self.root),'configure','--repo',str(self.root)])

    def test_metal_diagnostic_accepts_only_known_failure(self):
        path = self.root/'unsuppressed.log'
        known = 'third_party/metal-cpp/Foundation/NSObject.hpp: runtime error: member call on null pointer'
        path.write_text(known)
        record = {'log':path.name,'log_sha256':sha(path),'exit':-signal.SIGABRT}
        for code in (1, -signal.SIGABRT):
            record['exit'] = code
            gates.validate_metal_diagnostic(self.session, record)
        record['exit'] = -signal.SIGSEGV
        with self.assertRaisesRegex(RuntimeError, 'Unexpected unsuppressed'):
            gates.validate_metal_diagnostic(self.session, record)
        path.write_text('unrelated failure')
        record.update(exit=-signal.SIGABRT,log_sha256=sha(path))
        with self.assertRaisesRegex(RuntimeError, 'Unexpected unsuppressed'):
            gates.validate_metal_diagnostic(self.session, record)

    def test_incomplete_output_directories_are_retained(self):
        first = protocol.attempt_directory('retained')
        (first/'partial.jxl').write_bytes(b'partial')
        second = protocol.attempt_directory('retained')
        self.assertNotEqual(first,second)
        self.assertEqual((first/'partial.jxl').read_bytes(),b'partial')


if __name__ == '__main__':
    unittest.main()
