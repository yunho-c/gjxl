#!/usr/bin/env python3
"""Summarize complete 65-image rate routes with their quality boundaries."""
from pathlib import Path
import statistics

import study


def main():
    dc=Path('build/e3-e4-dc-grid-confirm-20260914').resolve()
    writer=Path('build/e4-native-tail-full-20260914').resolve()
    assert study.read(dc/'audit.json')['status']==study.read(writer/'audit.json')['status']=='passed'
    baseline=study.read(dc/'analysis.json');tail=study.read(writer/'analysis.json')
    manifest=study.read(writer/'manifest.json')
    scopes=['all','kodak_0_4mp','clic_1_8_to_3_4mp','12mp','24mp','48mp']
    names=['All 65','Kodak (24)','CLIC (32)','12 MP (3)','24 MP (3)','48 MP (3)']
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,5.5),layout='constrained',sharex=True)
    for ax,e in zip(axes,[3,4]):
        for label,offset,color,prefix in [('Native GJXL',-.18,'#888888','native'),
            ('Finer ordinary DC; changed pixels',.03,'#9966AA','candidate')]:
            vals=[]
            for scope in scopes:
                row=next(r for r in baseline['summary'] if r['range']==[75,85] and r['scope']==scope and r['comparison']==f'{prefix}-e{e}-vs-libjxl-e{e}')
                assert row['ready']==row['expected'];vals.append(row['pchip'])
            ax.scatter(vals,np.arange(6)+offset,color=color,label=label,s=35)
        if e==4:
            vals=[]
            for scope in scopes:
                selected=[r for r in tail['images'] if r['comparison']=='tail-vs-stock' and
                    (scope=='all' or manifest['images'][r['image_id']]['resolution_class']==scope)]
                expected=sum(scope=='all' or im['resolution_class']==scope for im in manifest['images'].values())
                assert len(selected)==expected and all(r['status']=='ready' for r in selected)
                vals.append(statistics.mean(r['pchip'] for r in selected))
            ax.scatter(vals,np.arange(6)+.24,color='#228866',marker='D',label='Libjxl writer; identical native pixels',s=35)
        ax.set(yticks=np.arange(6),yticklabels=names,xlabel=f'BD-rate vs libjxl effort {e} (%)',title=f'Effort {e}')
        ax.axvline(0,color='#444444',lw=.8);ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    h,l=axes[1].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncols=1,frameon=False)
    fig.suptitle('Measured routes past the same-effort gap · SSIMULACRA2 75–85\nAll 65 original inputs; no extrapolation · latency and production defaults unqualified',fontsize=12)
    dest=Path('build/e3-e4-rate-20260914')
    for ext in ['png','svg']:fig.savefig(dest/f'final-routes.{ext}',dpi=180)
    print(dest/'final-routes.png')


if __name__=='__main__':main()
