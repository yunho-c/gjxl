"""Explicit inputs, immutable reconstruction, command retention and identities."""
from pathlib import Path
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import tarfile
import time

PACKAGE = Path(__file__).resolve().parent
BASELINE = 'ec4d4c5317d983b5f6df29d2474923443c9cfe63'
HISTORICAL = '07dd92e'
DECODER_REVISION = 'e8ff09762481785938d8e4e01333ed3917571161'


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}


def package_hashes():
    # README describes the run; executable/configuration inputs define identity.
    return {p.name: sha(p) for p in sorted(PACKAGE.iterdir())
            if p.suffix in ('.py', '.cpp', '.json', '.supp') or p.name == 'CMakeLists.txt'}


def output(command, cwd=None):
    return subprocess.check_output(list(map(str, command)), cwd=cwd, text=True).strip()


def validate_corpus(directory):
    manifest = read(PACKAGE / 'corpus.json')
    for row in manifest['inputs']:
        path = directory / row['canonical_path']
        check(path.is_file(), f'Missing canonical input: {path}')
        check(sha(path) == row['canonical_sha256'], f'Canonical input hash mismatch: {path}')
    return {row['canonical_path']: row['canonical_sha256'] for row in manifest['inputs']}


def configure(args):
    check(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Qualification requires Apple Silicon macOS')
    repository = args.repo.resolve()
    candidate = HISTORICAL if args.historical else args.candidate_ref
    refs = {label: output(['git', 'rev-parse', ref + '^{commit}'], repository)
            for label, ref in [('integrated', args.baseline_ref), ('candidate', candidate)]}
    corpus = args.corpus.resolve()
    inputs = validate_corpus(corpus)
    decoder, info = args.decoder.resolve(), args.info.resolve()
    check(decoder.is_file() and info.is_file(), 'Supply both djxl and jxlinfo')
    version = subprocess.run([str(decoder), '--version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True).stdout
    check(DECODER_REVISION[:8] in version, 'djxl must be built from pinned libjxl ' + DECODER_REVISION)
    compiler = output(['xcrun', '--find', 'clang++'])
    config = {'schema': 1, 'repo': str(repository), 'revisions': refs,
              'capabilities': {'integrated': args.baseline_capability, 'candidate': 'current'},
              'corpus': str(corpus), 'inputs': inputs, 'decoder': str(decoder), 'info': str(info),
              'decoder_sha256': sha(decoder), 'info_sha256': sha(info),
              'decoder_revision': DECODER_REVISION, 'decoder_version': version,
              'compiler': compiler, 'compiler_version': output([compiler, '--version']),
              'sdk': output(['xcrun', '--show-sdk-path']), 'sdk_version': output(['xcrun', '--show-sdk-version']),
              'jobs': args.jobs, 'profile': args.profile}
    destination = args.out.resolve()
    check(destination != repository and destination not in repository.parents and
          (not destination.is_relative_to(repository) or destination.is_relative_to(repository / 'build')), 'Use a dedicated output directory, outside sources or beneath build/')
    if (destination / 'config.json').exists():
        check(read(destination / 'config.json') == config, 'Configuration changed; use a new output directory')
    else:
        check(not destination.exists() or not any(destination.iterdir()), 'Output directory is not empty')
        destination.mkdir(parents=True, exist_ok=True)
        save(destination / 'config.json', config)
    print(destination / 'config.json', flush=True)


def archive(repository, revision, destination):
    destination.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(['git', '-C', str(repository), 'archive', revision])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive_file:
        for entry in archive_file.getmembers():
            check(not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts, 'Unsafe archive path')
            if entry.issym() or entry.islnk():
                target = (destination / entry.name).parent / entry.linkname
                check(target.resolve().is_relative_to(destination.resolve()), 'Archive link escapes source tree')
        archive_file.extractall(destination, filter='data')


class Session:
    def __init__(self, directory):
        self.out = Path(directory).resolve()
        self.config = read(self.out / 'config.json')
        self.repo = Path(self.config['repo'])
        self.sources = {label: self.out / 'sources' / label for label in self.config['revisions']}
        self.build_roots = {label: self.out / 'builds' / label for label in self.sources}
        self.builds = {label: p / 'codec' for label, p in self.build_roots.items()}
        self.corpus = Path(self.config['corpus'])
        self.decoder = Path(self.config['decoder'])
        self.info = Path(self.config['info'])

    def event(self, kind, **values):
        with (self.out / 'execution-events.jsonl').open('a') as stream:
            stream.write(json.dumps({'event': kind, 'time': time.time(), **values}) + '\n')

    def unique(self, stem, suffix):
        attempt = 0
        while True:
            path = self.out / (stem + (f'-attempt{attempt}' if attempt else '') + suffix)
            if not path.exists():
                return path
            attempt += 1

    def run(self, command, stem, expected=(0,), environment=None, timeout=1200):
        command = list(map(str, command))
        log = self.unique(stem, '.log')
        started = time.time()
        # Tests also invoke Clang directly, outside CMake's -isysroot flags.
        # Keep those subprocesses on the same SDK selected at configuration.
        configured_environment = {'SDKROOT': self.config['sdk']}
        with log.open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                cwd=self.repo, env={**os.environ, **configured_environment, **(environment or {})}, start_new_session=True)
            self.event('started', command=command, pid=process.pid, log=log.name)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                self.event('interrupted', command=command, pid=process.pid, log=log.name)
                raise
        record = {'command': command, 'log': log.name, 'log_sha256': sha(log), 'exit': code,
                  'started': started, 'ended': time.time()}
        self.event('completed' if code in expected else 'failed', **record)
        check(code in expected, f'Command failed ({code}); retained {log}')
        return record

    def reconstruct(self, label):
        source = self.sources[label]
        stamp = self.out / (label + '-source.json')
        if stamp.exists():
            check(hashes(source) == read(stamp)['hashes'], f'Reconstructed {label} source changed')
            return
        check(not source.exists(), f'Incomplete source reconstruction retained: {source}; use a new output directory')
        revision = self.config['revisions'][label]
        archive(self.repo, revision, source)
        # Only metal-cpp is required for these GJXL builds. The optional libjxl
        # source is an independent, explicitly provisioned decoder dependency.
        entry = output(['git', 'ls-tree', revision, 'third_party/metal-cpp'], self.repo)
        metal_revision = entry.split()[2]
        metal_repo = self.repo / 'third_party/metal-cpp'
        present = subprocess.run(['git', '-C', str(metal_repo), 'cat-file', '-e', metal_revision], capture_output=True).returncode == 0
        if not present:
            metal_repo = self.out / 'metal-cpp-repository'
            if not metal_repo.exists():
                self.run(['git', 'clone', '--no-checkout', 'https://github.com/apple/metal-cpp.git', metal_repo], 'metal-cpp-clone')
            self.run(['git', '-C', metal_repo, 'fetch', 'origin', metal_revision], 'metal-cpp-fetch')
        archive(metal_repo, metal_revision, source / 'third_party/metal-cpp')
        save(stamp, {'revision': revision, 'metal_cpp_revision': metal_revision, 'hashes': hashes(source)})

    def compile(self, label, name=None, sanitizer=None):
        name = name or label
        directory = self.out / 'builds' / name
        capability = self.config['capabilities'][label]
        check((self.sources[label] / 'src/codestream/workflow_admission.h').is_file() == (capability == 'current'),
              'Declared driver capability does not match source revision: ' + label)
        cxx = '' if not sanitizer else '-fsanitize=' + sanitizer + ' -fno-omit-frame-pointer'
        objcxx = cxx if sanitizer == 'thread' else ''
        self.run(['cmake', '-S', PACKAGE, '-B', directory, '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_STANDARD=20',
            '-DCMAKE_CXX_COMPILER=' + self.config['compiler'], '-DCMAKE_OBJCXX_COMPILER=' + self.config['compiler'],
            '-DCMAKE_OSX_SYSROOT=' + self.config['sdk'], '-DCMAKE_OSX_ARCHITECTURES=arm64',
            '-DCMAKE_CXX_FLAGS=' + cxx, '-DCMAKE_OBJCXX_FLAGS=' + objcxx,
            '-DGJXL_QUALIFICATION_SOURCE=' + str(self.sources[label]),
            '-DGJXL_QUALIFICATION_CAPABILITY=' + capability,
            '-DGJXL_BUILD_TESTS=ON', '-DGJXL_BUILD_BENCHMARKS=' + ('OFF' if sanitizer else 'ON'),
            '-DGJXL_ENABLE_LIBJXL_REFERENCE=OFF', '-DGJXL_ENABLE_METAL_PROFILING=OFF'], name + '-configure')
        # Order the historical producer explicitly before embedding/linking.
        self.run(['cmake', '--build', directory, '--target', 'gjxl_metal_shaders', '-j', self.config['jobs']], name + '-shaders')
        self.run(['cmake', '--build', directory, '-j', self.config['jobs']], name + '-build', timeout=1800)
        env = {'ASAN_OPTIONS': 'detect_leaks=0:halt_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1'} if sanitizer else None
        self.run([directory / 'gjxl_qualification_embedding', directory / 'codec/metal/gjxl.metallib'],
                 name + '-embedding', environment=env)
        return directory

    def build(self):
        if (self.out / 'build.json').exists():
            self.identities()
            print('Matching build identities retained.', flush=True)
            return
        for label in self.sources:
            self.reconstruct(label)
            directory = self.compile(label)
            # Stable driver names are part of the historical job protocol.
            import shutil
            shutil.copy2(directory / 'gjxl_qualification_driver', self.out / (label + '-driver'))
        record = {'status': 'complete', 'configuration': self.config, 'package': package_hashes(),
                  'candidate_revision': self.config['revisions']['candidate'],
                  'integrated_revision': self.config['revisions']['integrated'],
                  'sources': {k: read(self.out / (k + '-source.json')) for k in self.sources},
                  'drivers': {k: sha(self.out / (k + '-driver')) for k in self.sources},
                  'artifacts': {}, 'machine': {'platform': platform.platform(),
                    'os': output(['sw_vers']), 'model': output(['sysctl', '-n', 'hw.model']),
                    'memory': output(['sysctl', '-n', 'hw.memsize']), 'cpus': output(['sysctl', '-n', 'hw.logicalcpu'])}}
        for directory in self.build_roots.values():
            for p in directory.rglob('*'):
                if p.is_file() and (p.suffix in ('.a', '.metallib') or p.name == 'CMakeCache.txt' or
                                   (p.name.startswith('gjxl_') and os.access(p, os.X_OK))):
                    record['artifacts'][str(p.relative_to(self.out))] = sha(p)
        check(sha(self.builds['integrated'] / 'metal/gjxl.metallib') == sha(self.builds['candidate'] / 'metal/gjxl.metallib'),
              'Protocol requires matching shader payloads; use a separately reviewed protocol for kernel changes')
        save(self.out / 'build.json', record)

    def identities(self):
        record = read(self.out / 'build.json')
        check(record['configuration'] == self.config, 'Build configuration changed')
        check(record['package'] == package_hashes(), 'Qualification package changed; use a new run')
        for label, source in self.sources.items():
            check(hashes(source) == record['sources'][label]['hashes'], f'{label} source changed')
        for label, digest in record['drivers'].items():
            check(sha(self.out / (label + '-driver')) == digest, f'{label} driver changed')
        for name, digest in record['artifacts'].items():
            check(sha(self.out / name) == digest, f'Build artifact changed: {name}')
        check(validate_corpus(self.corpus) == self.config['inputs'], 'Corpus changed')
        check(sha(self.decoder) == self.config['decoder_sha256'] and sha(self.info) == self.config['info_sha256'], 'Decoder tools changed')
        return record

    def quiet(self):
        state = output(['ps', '-axo', 'pid,pcpu,etime,command'])
        executables = {}
        for line in output(['ps', '-axo', 'pid=,comm=']).splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) == 2: executables[int(fields[0])] = Path(fields[1]).name
        conflicts = []
        for line in state.splitlines()[1:]:
            fields = line.strip().split(None, 3)
            if len(fields) != 4 or int(fields[0]) == os.getpid():
                continue
            command = fields[3]
            name = executables.get(int(fields[0]), '')
            if (name.startswith('gjxl_') or name.endswith('-driver') or name in
                ('ctest', 'clang', 'clang++', 'cc1', 'c++', 'ninja', 'make', 'gmake') or
                (name == 'cmake' and '--build' in command)):
                conflicts.append(line)
        check(not conflicts, 'Measured run requires a quiet encoder/build environment:\n' + '\n'.join(conflicts))
        return state
