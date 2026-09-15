#!/usr/bin/env python3
"""Scientific native-pixel reconstruction panels; display transform is not a metric input."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

import cross_metric
import study

ROOT=cross_metric.ROOT
CROPS={
    'clic2024_test/b51d5fb537246482ef7a7ab63f093cab':('face',.48,.45,384),
    'clic2024_test/d1a9be98d1936065967adac50a6fb750':('smooth-background',.87,.52,384),
    'clic2024_test/097cb426910ba8ce2525dd8bb7fb1777':('dark-background',.87,.46,384),
    'kodak/13':('water-and-stones',.48,.70,256),
    'unsplash/campus_interior/48mp':('concrete-wall',.28,.72,384),
    'unsplash/alpine_lake/24mp':('water-gradient',.80,.91,384),
}


def crop_pfm(path,box):
    with Path(path).open('rb') as f:
        assert f.readline().strip()==b'PF'
        width,height=map(int,f.readline().split());scale=float(f.readline());offset=f.tell()
    pixels=np.memmap(path,dtype='<f4' if scale<0 else '>f4',mode='r',offset=offset,shape=(height,width,3))
    x,y,size=box
    linear=np.maximum(0,np.array(pixels[::-1][y:y+size,x:x+size])*abs(scale))
    del pixels
    srgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
    return Image.fromarray(np.uint8(np.clip(srgb,0,1)*255+.5))


def main():
    manifest=study.read(ROOT/'manifest.json')
    data={(r['image_id'],r['arm']):r for r in study.rows(ROOT/'matches.jsonl')}
    dest=ROOT/'visual';dest.mkdir(exist_ok=True)
    for name,(label,fx,fy,size) in CROPS.items():
        if not all((name,arm) in data for arm in manifest['arms']):continue
        image=manifest['images'][name]
        x=max(0,min(image['width']-size,round(fx*image['width']-size/2)))
        y=max(0,min(image['height']-size,round(fy*image['height']-size/2)))
        box=(x,y,size);key=cross_metric.identifier(name,label)
        records={arm:data[name,arm] for arm in manifest['arms']}
        identity={'image':name,'crop':[x,y,size,size],'reference_sha256':image['pfm_sha256'],
                  'records':records,'display':'linear-sRGB to clipped 8-bit sRGB; native pixels, no resampling; not metric input',
                  'selection':'Content inspection guided by largest Butteraugli disagreement and smooth areas; illustrative, not blinded perceptual validation'}
        record_path=dest/f'{key}.json';png=dest/f'{key}.png'
        if record_path.exists() and study.read(record_path).get('identity')==identity:
            assert study.sha(png)==study.read(record_path)['png_sha256'];continue
        assert study.sha(image['pfm_path'])==image['pfm_sha256']
        reference=crop_pfm(image['pfm_path'],box);panels={}
        for arm,row in records.items():
            assert study.sha(row['output_path'])==row['output_sha256']
            decoded=dest/'temporary.pfm'
            result=subprocess.run([manifest['djxl'],row['output_path'],str(decoded),
                '--color_space=RGB_D65_SRG_Rel_Lin','--num_threads=1'],capture_output=True,text=True)
            study.append(dest/'commands.jsonl',{'argv':result.args,'returncode':result.returncode,'stderr':result.stderr})
            result.check_returncode();assert study.sha(decoded)==row['decoded_sha256']
            panels[arm]=crop_pfm(decoded,box);decoded.unlink()
        pad=12;header=55;top=56;out=Image.new('RGB',(4*(size+pad)+pad,2*(size+header+pad)+top),'white');draw=ImageDraw.Draw(out)
        draw.text((pad,10),f'{name} | {label} | crop ({x}, {y}, {size}, {size}) | native pixels',fill='black')
        draw.text((pad,27),'Scores are actual measurements. UNRESOLVED targets are not matched-quality evidence. Lower BA means less error.',fill='black')
        for r,e in enumerate((3,4)):
            arms=[None,study.native(e),f'e{e}-round-smooth0-precision1',f'libjxl-e{e}']
            for c,arm in enumerate(arms):
                xx=pad+c*(size+pad);yy=top+r*(size+header+pad)
                if arm is None:
                    tile=reference;text='Original reference'
                else:
                    row=records[arm];tile=panels[arm]
                    text=(f'{arm}\nS2={row["score"]:.4f}   BA={row["butteraugli"]:.4f}\n'
                          f'{row["encoded_bytes"]:,} bytes'+('   UNRESOLVED' if not row['matched'] else ''))
                draw.multiline_text((xx,yy),text,fill='black',spacing=3)
                out.paste(tile,(xx,yy+header))
        out.save(png);study.save(record_path,{'identity':identity,'png_sha256':study.sha(png)})
        print(label,png,flush=True)
    entries=[]
    for p in sorted(dest.glob('*.json')):
        if p.name=='index.json':continue
        record=study.read(p);entries.append({'path':str(p.with_suffix('.png')),'image':record['identity']['image'],
            'crop':record['identity']['crop'],'sha256':record['png_sha256']})
    study.save(dest/'index.json',{'panels':entries,'count':len(entries)})


if __name__=='__main__':main()
