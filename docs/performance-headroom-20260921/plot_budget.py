"""Plot saved-data attribution and explicitly hypothetical Amdahl sensitivities."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
rows = {(r['resolution_class'], int(r['effort'])): r
        for r in csv.DictReader((HERE / 'summary.csv').open())}
cases = [('24mp', 1), ('24mp', 7), ('24mp', 10), ('48mp', 7), ('48mp', 10),
         ('clic_1_8_to_3_4mp', 10)]
labels = ['24 MP · effort 1', '24 MP · effort 7', '24 MP · effort 10',
          '48 MP · effort 7', '48 MP · effort 10', 'CLIC 3.1–3.4 MP · effort 10']
categories = ['GPU perceptual work', 'GPU AC search', 'Other timed GPU work',
              'CPU serialization', 'Pipeline orchestration / gaps', 'Other time']
colors = ['#286b8b', '#50a3ad', '#a4c4cb', '#bb6349', '#9d988f', '#dedbd5']
values = []
for key in cases:
    r = rows[key]
    p, ac, gpu, ser, gap = [float(r[k + '_percent']) for k in
                            ['perceptual', 'ac_search', 'all_gpu', 'serializer', 'orchestration']]
    values.append([p, ac, gpu-p-ac, ser, gap, 100-gpu-ser-gap])
values = np.array(values)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={'width_ratios':[1.8,1.1]})
y = np.arange(len(cases)); left = np.zeros(len(cases))
for j, (cat, color) in enumerate(zip(categories, colors)):
    ax.barh(y, values[:,j], left=left, label=cat, color=color, height=.67)
    for i,v in enumerate(values[:,j]):
        if v >= 9:
            ax.text(left[i]+v/2, i, f'{v:.0f}%', ha='center', va='center',
                    color='white' if j in (0,1,3) else '#202020', fontsize=9)
    left += values[:,j]
ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0,100)
ax.set_xlabel('Share of the complete profiled API call (%)')
ax.set_title('The main cost changes with effort', loc='left', weight='bold', pad=15)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
handles, legend_labels = ax.get_legend_handles_labels()
fig.legend(handles, legend_labels, ncol=3, loc='lower left', bbox_to_anchor=(.05,.115), frameon=False, fontsize=9)
r = rows['24mp',7]
targets = ['perceptual','ac_search','serializer','all_gpu']
gains = [float(r[k+'_percent'])/2 for k in targets]
names = ['Perceptual work 2× faster','AC search 2× faster','Serialization 2× faster','All timed GPU work 2× faster']
bx.barh(np.arange(4), gains, color=[colors[0],colors[1],colors[3],'#25495c'],height=.57)
for i,v in enumerate(gains):bx.text(v+.5,i,f'{v:.1f}%',va='center',fontsize=10)
bx.set_yticks(np.arange(4),names,fontsize=9);bx.invert_yaxis();bx.set_xlim(0,43)
bx.set_xlabel('Hypothetical end-to-end\nlatency reduction (%)',fontsize=9)
bx.set_title('What faster stages would buy\n24 MP, effort 7 · independent scenarios',loc='left',weight='bold',pad=15,fontsize=10)
bx.spines['left'].set_visible(False);bx.tick_params(axis='y',length=0)
fig.suptitle('GJXL performance headroom · M4 Pro',x=.05,ha='left',fontsize=17,weight='bold',y=.985)
fig.text(.05,.915,'Frozen 4f3e414 study · nominal Q80 / distance 1.9 · 8 CPU participants · same-call host/GPU attribution',fontsize=10,color='#555555')
fig.subplots_adjust(left=.16,right=.97,top=.78,bottom=.30,wspace=.78)
fig.text(.05,.035,'Sensitivity calculations are not demonstrated gains. Six selected images; six paired repetitions per setting.\n'
         'GPU intervals are measured; orchestration/gaps are unresolved elapsed time. '
         'Perceptual work includes reference features and Butteraugli comparison. Shares use profiled totals; production latency uses ordinary calls.',
         fontsize=8.5,color='#555555')
for ext in ('png','svg'):
    fig.savefig(HERE/f'performance-budget.{ext}',dpi=180)
