"""Reports derived from this workspace's actual outputs; no embedded study results."""
from pathlib import Path
import json
import pandas as pd
from .io import development,sha,versions,write_json


def create_report(c):
    root=c['_root'];result=root/'private/results';m,_=development(root)
    lines=['# AD-NC analysis run','',f'Development subjects: {len(m)}; AD={sum(m.group=="AD")}; NC={sum(m.group=="NC")}.',
           '','Full EEG, PCA and cPCA are parallel representations with regularized logistic regression.']
    for name in ['internal_metric_summary.csv','external_metrics.csv','external_model_comparisons.csv']:
        p=result/name
        if p.exists():lines+=['',f'## {name}','','```text',pd.read_csv(p).to_string(index=False),'```']
        else:lines+=['',f'{name}: not run in this workspace.']
    frozen=Path(c['paths']['frozen_bundle'])/'freeze_manifest.json'
    if frozen.exists():
        mf=json.loads(frozen.read_text(encoding='utf-8'));lines+=['','## Frozen parameters','',json.dumps(mf['selected_parameters'],indent=2)]
    lines+=['','## Interpretation limits','','External data are transformed with frozen models; no external tuning. Alpha=0 does not establish an extra background contrast effect. Model weights, training Haufe patterns, test permutation dependence and AD-minus-NC effects answer different questions. Sixteen scalp electrodes support coarse scalp patterns only. Pooled MMSE association is not evidence of AD-only severity. Synthetic runs are software demonstrations, not paper results.']
    (result/'run_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(result/'report_manifest.json',{'versions':versions(),'output_sha256':{p.name:sha(p) for p in result.glob('*.csv')}})
    print('[report] private/results/run_report.md written',flush=True)
