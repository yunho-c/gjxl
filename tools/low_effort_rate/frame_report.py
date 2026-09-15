#!/usr/bin/env python3
"""Additive native bit-budget comparisons; never infer rate from fixed distance alone."""
from collections import Counter
import statistics

import cross_metric
import frame_accounting
import study


def main():
    root=frame_accounting.ROOT
    matches={(r['image_id'],r['arm']):r for r in study.rows(cross_metric.ROOT/'matches.jsonl')}
    all_rows=[];summaries=[]
    fingerprint_keys=['raw_quant_fingerprint','sharpness_fingerprint','ytox_fingerprint',
        'ytob_fingerprint','ac_coefficient_fingerprints','global_scale','quant_dc','ac_section_bits']
    for mode in ['fixed','matched']:
        dest=root/mode
        if not (dest/'audit.json').exists():continue
        assert study.read(dest/'audit.json')['status']=='passed'
        data={(r['image_id'],r['arm']):r for r in study.rows(dest/'observations.jsonl')}
        for e in [3,4]:
            native=study.native(e);candidate=f'e{e}-round-smooth0-precision1';pairs=[]
            for name in cross_metric.IMAGES:
                a=data[name,native];b=data[name,candidate]
                ac=a['capture'];bc=b['capture'];baseline=ac['total_bytes']*8
                accepted=True
                if mode=='matched':
                    ma=matches[name,native];mb=matches[name,candidate]
                    accepted=ma['matched'] and mb['matched'] and abs(ma['score']-mb['score'])<=.05
                components={k:100*(bc[k]-ac[k])/baseline for k in frame_accounting.COMPONENTS}
                total=100*(bc['total_bytes']/ac['total_bytes']-1)
                assert abs(sum(components.values())-total)<1e-10
                row={'image_id':name,'effort':e,'mode':mode,'accepted':accepted,
                    'score_difference':b['score']-a['score'],'byte_change_percent':total,
                    'component_contributions_pp':components,
                    'same_state':{k:ac[k]==bc[k] for k in fingerprint_keys},
                    'dc_token_delta_bits':bc['dc_token_bits']-ac['dc_token_bits'],
                    'ac_section_delta_bits':bc['ac_section_bits']-ac['ac_section_bits']}
                all_rows.append(row)
                if accepted:pairs.append(row)
            summary={'mode':mode,'effort':e,'accepted':len(pairs),'expected':12,
                'aggregation':'arithmetic mean of per-image deltas normalized to each native file; component means sum to mean byte change',
                'scope':'all selected inputs' if len(pairs)==12 else 'accepted matched subset only'}
            if pairs:
                summary.update(mean_score_difference=statistics.mean(r['score_difference'] for r in pairs),
                    mean_byte_change_percent=statistics.mean(r['byte_change_percent'] for r in pairs),
                    mean_component_contributions_pp={k:statistics.mean(r['component_contributions_pp'][k] for r in pairs) for k in frame_accounting.COMPONENTS},
                    state_equal_counts={k:sum(r['same_state'][k] for r in pairs) for k in fingerprint_keys},
                    dc_token_increase=sum(r['dc_token_delta_bits']>0 for r in pairs),
                    ac_section_decrease=sum(r['ac_section_delta_bits']<0 for r in pairs))
            summaries.append(summary);print(summary)
    study.save(root/'analysis.json',{'summary':summaries,'images':all_rows,
        'limits':'Native bit allocation, not libjxl same-frame writer attribution. Fixed distance comparisons are not quality matched. Hash fingerprints are diagnostic; codestream identity is SHA256 checked.'})


if __name__=='__main__':main()
