# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import subprocess,sys,struct
encoder,oracle,root=sys.argv[1:]
root=Path(root);root.mkdir(parents=True,exist_ok=True)
cases=0
for channels in [1,3,4]:
 for bits in [8,16]:
  w,h=257,3
  samples=[(0 if c==3 and x%3==0 else (x*313+y*17+c*71))&((1<<bits)-1) for y in range(h) for x in range(w) for c in range(channels)]
  raw=root/f'{channels}-{bits}.raw'
  raw.write_bytes(b''.join(struct.pack('<H',v) for v in samples))
  for pam in [False,True]:
   if channels==4 and not pam:continue
   path=root/f'{channels}-{bits}-{pam}.pnm'
   if pam:
    header=f'P7\nWIDTH {w}\nHEIGHT {h}\nDEPTH {channels}\nMAXVAL {(1<<bits)-1}\nTUPLTYPE { {1:"GRAYSCALE",3:"RGB",4:"RGB_ALPHA"}[channels]}\nENDHDR\n'
   else:header=f'P{5 if channels==1 else 6}\n# explicit source sRGB\n{w} {h}\n{(1<<bits)-1}\n'
   path.write_bytes(header.encode()+ (bytes(samples) if bits==8 else b''.join(struct.pack('>H',v) for v in samples)))
   for entropy in ['prefix','ans']:
    output=path.with_suffix('.'+entropy+'.jxl')
    subprocess.run([encoder,'--modular','--input-color-space','srgb','--threads','4','--search','--entropy',entropy,str(path),str(output)],check=True,capture_output=True)
    subprocess.run([oracle,'--check-file',str(output),str(raw),str(w),str(h),str(channels),str(bits)],check=True,capture_output=True)
    cases+=1
valid=root/'1-8-False.pnm'; output=root/'atomic.jxl'
invalids=[b'P5\n1 1\n254\n\0',b'P6\n2 2\n255\n\0',b'PF\n1 1\n-1\n'+bytes(12),
          b'P7\nWIDTH 1\nHEIGHT 1\nDEPTH 2\nMAXVAL 255\nTUPLTYPE GRAYSCALE_ALPHA\nENDHDR\n\0\0',
          b'P7\nWIDTH 1\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nTUPLTYPE RGB\nCOLORSPACE LINEAR\nENDHDR\n\0\0\0',
          b'P5\n4294967296 1\n255\n\0', b'P5\n1 1\n255\n\0extra']
for i,data in enumerate(invalids):
 bad=root/f'bad-{i}.pnm';bad.write_bytes(data);output.write_bytes(b'preserve')
 r=subprocess.run([encoder,'--modular','--input-color-space','srgb',str(bad),str(output)],capture_output=True)
 assert r.returncode!=0 and output.read_bytes()==b'preserve'
for flags in [[],['--input-color-space','linear'],['--input-color-space','srgb','--distance','1'],['--input-color-space','srgb','--backend','cuda'],['--input-color-space','srgb','--threads','257']]:
 output.write_bytes(b'preserve')
 r=subprocess.run([encoder,'--modular',*flags,str(valid),str(output)],capture_output=True)
 assert r.returncode!=0 and output.read_bytes()==b'preserve'
print(f'{cases} CLI files decoded exactly; {len(invalids)+5} invalid requests preserved output')
