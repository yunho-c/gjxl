# Regenerating the two synthetic canonical inputs

Run this from the GJXL checkout on the qualification toolchain. Replace the
canonical directory argument. The recipe extracts the exact `FillSynthetic`
function from the tracked driver, builds a small standalone exporter, and checks
both output hashes from `corpus.json`. It needs no GJXL build or external images.
Existing files are verified and never overwritten. Temporary compiler artifacts
are removed when the recipe exits.

```sh
python3 - /path/to/canonical <<'PY'
from pathlib import Path
import hashlib, json, subprocess, sys, tempfile

package = Path('tools/resident_qualification')
corpus = Path(sys.argv[1])
corpus.mkdir(parents=True, exist_ok=True)
source = (package / 'driver.cpp').read_text()
fill = source[source.index('void FillSynthetic('):source.index('\nvoid RequireStatus(')]
exporter = r'''#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
struct ImageStorage {
  struct { size_t width, height; } extent;
  std::array<std::vector<float>, 3> plane;
};
''' + fill + r'''
int main(int argc, char** argv) {
  if (argc != 4) return 1;
  const size_t w = std::stoull(argv[1]), h = std::stoull(argv[2]);
  ImageStorage image{{w, h}, {}};
  for (auto& p : image.plane) p.resize(w * h);
  FillSynthetic(&image);
  std::ofstream out(argv[3], std::ios::binary);
  out << "PF\n" << w << " " << h << "\n-1.0\n";
  for (size_t y = h; y-- > 0;)
    for (size_t x = 0; x < w; ++x)
      for (const auto& p : image.plane)
        out.write(reinterpret_cast<const char*>(&p[y*w+x]), sizeof(float));
  return out ? 0 : 1;
}
'''
with tempfile.TemporaryDirectory(prefix='gjxl-synthetic-') as temporary:
    root = Path(temporary)
    source, binary = root / 'export.cpp', root / 'export'
    source.write_text(exporter)
    sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
    subprocess.run(['xcrun', 'clang++', '-std=c++20', '-O3', '-isysroot', sdk,
                    str(source), '-o', str(binary)], check=True)
    for row in json.loads((package / 'corpus.json').read_text())['inputs']:
        if not row['canonical_path'].startswith('padded-stress-'):
            continue
        path = corpus / row['canonical_path']
        if not path.exists():
            subprocess.run([str(binary), str(row['width']), str(row['height']), str(path)], check=True)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != row['canonical_sha256']:
            raise RuntimeError('Canonical hash mismatch: ' + str(path))
        print(path.name, digest)
PY
```

The other 36 canonical PFMs require the declared natural-image sources and
ImageMagick conversions in [corpus.json](corpus.json); see the
[corpus prerequisites](README.md#canonical-corpus). Input hashes, rather than
filenames or visually similar images, define this qualification corpus.
