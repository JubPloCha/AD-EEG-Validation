from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from .io import sha, write_json, prepare_external, table
from .modeling import load_frozen_bundle, predict_frozen
from .workflow import log

def predict_frame(models, features, covariates, unit_token='EEGLAB_microvolt'):
    if unit_token!='EEGLAB_microvolt': raise ValueError('Signal units must be verified EEGLAB microvolts')
    names=next(iter(models.values()))['feature_names']
    if features.columns.tolist()!=names: raise ValueError('Feature names/order differ from frozen schema')
    if features.index.has_duplicates or covariates.index.has_duplicates or features.index.hasnans: raise ValueError('Missing/duplicate subject_id')
    if set(features.index)!=set(covariates.index): raise ValueError('Feature/covariate identity mismatch')
    md=covariates.loc[features.index]
    return predict_frozen(models,features.to_numpy(float),md.age.to_numpy(float),md.sex.to_numpy(float))


def predict_external(c):
    """Transform-only prediction; diagnosis and development data not required."""
    root=c['_root']; manifest,models=load_frozen_bundle(Path(c['paths']['frozen_bundle']))
    prepare_external(c, require_labels=False)
    m=table(root/'private/external/subjects.csv')
    f=table(root/'private/external/features.csv')
    # Reject overlap when development inputs are present in this workspace.
    dev=root/'private/development/subjects.csv'
    if dev.exists() and set(table(dev).index)&set(m.index):
        raise ValueError('External and development subject_id overlap')
    predictions=predict_frame(models,f,m)
    rows=[{'subject_id':sid,'model':name,'freeze_id':manifest['freeze_id'],
           'probability_ad':v['probability_ad'][i],'decision':v['decision'][i]}
          for name,v in predictions.items() for i,sid in enumerate(f.index)]
    pd.DataFrame(rows).to_csv(root/'private/results/external_transform_only_predictions.csv',index=False)
    log(root,'predict',f'Transform-only predictions for {len(m)} subjects; no fitting')

def metric_values(y,p):
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':roc_auc_score(y,p),'pr_auc_average_precision':average_precision_score(y,p),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'balanced_accuracy':balanced_accuracy_score(y,pred),'accuracy':accuracy_score(y,pred),'f1':f1_score(y,pred,zero_division=0),'brier':brier_score_loss(y,p),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def evaluate_external(c):
    root=c['_root']; folder=Path(c['paths']['frozen_bundle']);manifest,models=load_frozen_bundle(folder)
    readiness=root/'private/results/external_readiness.json'
    if not readiness.exists() or not json.loads(readiness.read_text())['ready']:
        raise RuntimeError('External reference verification unresolved; no external metrics are computed')
    protocol=json.loads((root/'private/evaluation_protocol.json').read_text(encoding='utf-8'))
    before={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    m=table(root/'private/external/subjects.csv').query('included == True')
    f=table(root/'private/external/features.csv').loc[m.index]
    if set(m.group)!={'AD','NC'}:raise RuntimeError('External primary evaluation requires both AD and NC; unresolved reference is not a QC exclusion')
    prediction=predict_frame(models,f,m);y=m.group.eq('AD').astype(int).to_numpy()
    # Identical fixed draws for all seven models and paired comparisons.
    rng=np.random.default_rng(protocol['bootstrap']['seed']);ib=np.flatnonzero(y==1);ic=np.flatnonzero(y==0)
    draws=np.array([np.r_[rng.choice(ib,len(ib),replace=True),rng.choice(ic,len(ic),replace=True)] for _ in range(protocol['bootstrap']['n'])])
    rows=[];metrics=[];boot={};checks=[]
    for name,v in prediction.items():
        p=v['probability_ad'];g=metric_values(y,p)
        boot[name]=np.array([roc_auc_score(y[d],p[d]) for d in draws]);lo,hi=np.quantile(boot[name],[.025,.975])
        metrics.append({'model':name,'freeze_id':manifest['freeze_id'],'dataset_id':c.get('external',{}).get('dataset_id','external'),'n':len(y),'n_ad':int(y.sum()),'n_nc':int((y==0).sum()),'threshold':.5,**g,'auc_ci_lower':lo,'auc_ci_upper':hi,'bootstrap_n':len(draws)})
        for j,sid in enumerate(f.index):rows.append({'model':name,'freeze_id':manifest['freeze_id'],'dataset_id':c.get('external',{}).get('dataset_id','external'),'subject_id':sid,'y_true':int(y[j]),'probability_ad':p[j],'decision':v['decision'][j],'eeg_logit':v['eeg_score'][j],'threshold':.5,'included':True})
    reverse=f.iloc[::-1];rc=predict_frame(models,reverse,m)
    ml=m.copy();ml['group']=np.random.default_rng(99).permutation(ml.group)
    lp=predict_frame(models,f,ml)
    for name,v in prediction.items():
        re=rc[name]['probability_ad'][::-1];err=float(np.max(abs(v['probability_ad']-re)));le=float(np.max(abs(v['probability_ad']-lp[name]['probability_ad'])))
        checks.append({'model':name,'row_reorder_max_error':err,'external_label_permutation_max_error':le,'pass':err<=1e-8 and le==0})
    after={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    if before!=after:raise RuntimeError('External evaluation mutated frozen files')
    write_json(root/'private/results/frozen_hash_integrity.json',{'before':before,'after':after,'unchanged':before==after})
    comp=[]
    metricdf=pd.DataFrame(metrics).set_index('model')
    for a,b in protocol['paired_comparisons']:
        delta=boot[a]-boot[b];lo,hi=np.quantile(delta,[.025,.975])
        comp.append({'model_a':a,'model_b':b,'auc_difference':metricdf.loc[a,'roc_auc']-metricdf.loc[b,'roc_auc'],'paired_ci_lower':lo,'paired_ci_upper':hi,'n':len(y),'bootstrap_n':len(draws),'same_subject_draws':True})
    out=root/'private/results';pd.DataFrame(rows).to_csv(out/'external_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'external_metrics.csv',index=False)
    pd.DataFrame(comp).to_csv(out/'external_model_comparisons.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'external_invariance_validation.csv',index=False)
    np.savez_compressed(out/'external_auc_bootstrap.npz',subject_draws=draws,**boot)
    # Predeclared age-range sensitivity: training NC support, never re-fit.
    tr=models['age_adjusted_eeg']['transform'];supported=m.age.between(tr['age_support_min'],tr['age_support_max']).to_numpy()
    pd.DataFrame([{'model':name,'n':int(supported.sum()),'age_min':tr['age_support_min'],'age_max':tr['age_support_max'],**(metric_values(y[supported],v['probability_ad'][supported]) if len(np.unique(y[supported]))==2 else {'roc_auc':np.nan,'status':'NA: fewer than two classes within training age support'})} for name,v in prediction.items()]).to_csv(out/'external_age_support_sensitivity.csv',index=False)
    if not pd.DataFrame(checks)['pass'].all():raise RuntimeError('External invariance validation failed')
    log(root,'external',f'Evaluated all seven frozen models on {len(m)} subjects; no fitting')
