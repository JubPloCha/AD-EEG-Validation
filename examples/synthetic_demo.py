"""Purely random full-schema software demo; never represents study results."""
from pathlib import Path
import argparse
import json
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from cpca_ad_nc.io import versions,write_json
from cpca_ad_nc.modeling import ModelConfig,select_hyperparameters,fit_frozen_models,save_frozen_bundle,load_frozen_bundle
from cpca_ad_nc.external import predict_frame


def generate(folder):
    """No real input is read: fixed-schema values all originate from RNG."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    names=json.loads((Path(__file__).resolve().parents[1]/'configs/feature_schema.json').read_text(encoding='utf-8'))['feature_order']
    rng=np.random.default_rng(20261018);data={}
    for cohort,n in [('development',96),('external',32)]:
        ids=[f'SYN_{cohort.upper()}_{i:03d}' for i in range(n)]
        y=np.tile([0,1],n//2);age=55+np.tile(np.arange(n//2)%20,2);sex=np.tile([0,0,1,1],n//4)
        X=rng.normal(size=(n,len(names)));X[:,:8]+=.5*y[:,None];X+=.005*(age-65)[:,None]
        f=pd.DataFrame(X,index=pd.Index(ids,name='subject_id'),columns=names)
        m=pd.DataFrame({'subject_id':ids,'group':np.where(y,'AD','NC'),'age':age,'sex':sex,'mmse':np.nan,'education_years':np.nan}).set_index('subject_id')
        f.to_csv(folder/f'{cohort}_features.csv');m.to_csv(folder/f'{cohort}_subjects.csv');data[cohort]=(m,f)
    return data


def run(folder):
    data=generate(folder);m,f=data['development'];me,fe=data['external']
    assert not set(m.index)&set(me.index)
    # Explicit small DEMO grid, not the paper's parameter search.
    cfg=ModelConfig(alpha_grid=(0.,1.),dim_grid=(1,2),c_grid=(.1,1.),inner_splits=3,outer_splits=2,outer_repeats=1)
    X=f.to_numpy();y=m.group.eq('AD').to_numpy(int);age=m.age.to_numpy(float);sex=m.sex.to_numpy(float)
    selected,table=select_hyperparameters(X,y,m.group.to_numpy(),age,sex,f.columns,cfg,cfg.random_seed)
    models=fit_frozen_models(X,y,m.group.to_numpy(),age,sex,f.columns,selected,cfg,cfg.random_seed)
    reference=predict_frame(models,fe,me)
    bundle=Path(folder)/'frozen_demo';save_frozen_bundle(bundle,models,selected,m.index.tolist(),{},cfg,versions())
    _,loaded=load_frozen_bundle(bundle);actual=predict_frame(loaded,fe,me)
    errors={k:float(np.max(abs(reference[k]['probability_ad']-actual[k]['probability_ad']))) for k in models}
    assert max(errors.values())<=1e-14
    try:predict_frame(loaded,fe.iloc[:,::-1],me)
    except ValueError:order_rejected=True
    else:raise AssertionError('Wrong feature order accepted')
    pd.DataFrame([{'subject_id':sid,'model':k,'probability_ad':v['probability_ad'][i]} for k,v in actual.items() for i,sid in enumerate(fe.index)]).to_csv(Path(folder)/'synthetic_predictions.csv',index=False)
    write_json(Path(folder)/'demo_validation.json',{'synthetic_only':True,'n_train':len(m),'n_external':len(me),'n_features':f.shape[1],'train_test_overlap':0,'feature_order_rejected':order_rejected,'max_reload_probability_error':max(errors.values())})
    print(f'SYNTHETIC DEMO ONLY: {len(m)} train / {len(me)} independent test; {f.shape[1]} features; 7 branches; reload error={max(errors.values()):.3g}; wrong order rejected.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):
        if args.output:run(args.output)
        else:
            with tempfile.TemporaryDirectory(prefix='SYN_EEG_') as d:run(Path(d))
