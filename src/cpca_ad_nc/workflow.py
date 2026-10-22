from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .io import development, fingerprint, sha, versions, write_json
from .modeling import ModelConfig, _inner_splits, fit_frozen_models, load_frozen_bundle, predict_frozen, save_frozen_bundle, select_hyperparameters


def model_config(c):
    return ModelConfig(**{k: tuple(v) if k.endswith('_grid') else v for k,v in c.get('model',{}).items()})


def source_hashes(root):
    return {p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('io.py'),Path(__file__).with_name('modeling.py'),root/'configs/feature_schema.json',root/'configs/interpretation_groups.json',root/'private/development/features.csv',root/'private/development/subjects.csv']}


def log(root, phase, message, **values):
    p=root/'runs'; p.mkdir(exist_ok=True)
    row={'utc':pd.Timestamp.now(tz='UTC').isoformat(),'phase':phase,'message':message,**values}
    with (p/'execution.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,default=str)+'\n')
    print(f"[{phase}] {message}",flush=True)


def internal(c):
    root=c['_root']; cfg=model_config(c); m,f=development(root)
    splits=pd.read_csv(root/'private/development/outer_splits.csv')
    pairs=pd.read_csv(root/'private/development/matched_pairs.csv',dtype={'a_subject_id':str,'b_subject_id':str})
    pair_map={sid:int(row.pair_id) for _,row in pairs.iterrows() for sid in [row.a_subject_id,row.b_subject_id]}
    all_predictions=[]; validations=[]; inner_ids=[]
    result=root/'private/results'; result.mkdir(exist_ok=True)
    for _,s in splits.iterrows():
        rep = int(s['repeat'])
        tick=time.monotonic(); tr0=json.loads(s.train_subject_ids);te0=json.loads(s.test_subject_ids)
        tr=tr0;te=te0
        if len(set(tr))!=len(tr) or len(set(te))!=len(te) or set(tr)&set(te): raise ValueError('Outer identity overlap')
        trpair=np.array([pair_map[x] for x in tr0]) if s.queue!='full' else None
        tepair=np.array([pair_map[x] for x in te0]) if s.queue!='full' else None
        if trpair is not None and set(trpair)&set(tepair): raise ValueError('Outer pair overlap')
        mt=m.loc[tr]; me=m.loc[te]
        Xt=f.loc[tr].to_numpy(float); Xe=f.loc[te].to_numpy(float)
        yt=mt.group.eq('AD').astype(int).to_numpy(); ye=me.group.eq('AD').astype(int).to_numpy()
        a=mt.age.to_numpy(float);sx=mt.sex.to_numpy(float)
        seed=cfg.random_seed+rep*1000+int(s.fold)
        folder=root/'private/fold_models'/s.queue/f'repeat_{rep:02d}_fold_{s.fold:02d}'
        key=fingerprint(source_hashes(root),asdict(cfg),tr,te,Xt,Xe,a,sx,mt.group.tolist(),me[['age','sex','group']].to_dict('list'),trpair,sha(root/'private/development/outer_splits.csv'),sha(root/'private/development/matched_pairs.csv'),versions())
        cache=folder/'cache.json'
        hit=cache.exists() and json.loads(cache.read_text())['key']==key
        if hit:
            _,models=load_frozen_bundle(folder/'models'); log(root,'internal',f'cache hit {s.queue} {rep}/{s.fold}',key=key)
            ident=json.loads((folder/'identity.json').read_text(encoding='utf-8'))
            if ident['train_subject_ids']!=tr or ident['test_subject_ids']!=te or ident['queue']!=s.queue or ident['repeat']!=rep or ident['fold']!=int(s.fold):
                raise ValueError('Cached fold identities do not match the saved split')
        else:
            if folder.exists(): raise RuntimeError(f'Stale model cache requires explicit new run directory: {folder}')
            selected,table=select_hyperparameters(Xt,yt,mt.group.to_numpy(),a,sx,f.columns,cfg,seed,trpair)
            folder.mkdir(parents=True)
            table.to_csv(folder/'inner_candidate_scores.csv',index=False)
            models=fit_frozen_models(Xt,yt,mt.group.to_numpy(),a,sx,f.columns,selected,cfg,cfg.random_seed+rep+int(s.fold))
            write_json(folder/'identity.json',{'queue':s.queue,'repeat':rep,'fold':int(s.fold),'train_subject_ids':tr,'test_subject_ids':te,'train_pair_ids':trpair.tolist() if trpair is not None else [],'test_pair_ids':tepair.tolist() if tepair is not None else []})
            save_frozen_bundle(folder/'models',models,selected,tr,source_hashes(root),cfg,versions())
            write_json(cache,{'key':key,'reason':'all seven branches selected from raw inner-training data; no external inputs'})
            log(root,'internal',f'retrained {s.queue} {rep}/{s.fold}',key=key,seconds=time.monotonic()-tick)
        for k,(it,iv) in enumerate(_inner_splits(yt,cfg.inner_splits,seed,trpair)):
            if set(np.array(tr)[it])&set(np.array(tr)[iv]): raise AssertionError('Inner subject overlap')
            if trpair is not None and set(trpair[it])&set(trpair[iv]): raise AssertionError('Inner pair overlap')
            inner_ids.append({'queue':s.queue,'repeat':rep,'outer_fold':s.fold,'inner_fold':k,'train_ids':json.dumps(np.array(tr)[it].tolist()),'valid_ids':json.dumps(np.array(tr)[iv].tolist()),'train_pair_ids':json.dumps(trpair[it].tolist() if trpair is not None else []),'valid_pair_ids':json.dumps(trpair[iv].tolist() if trpair is not None else [])})
        pred=predict_frozen(models,Xe,me.age.to_numpy(float),me.sex.to_numpy(float))
        # Drop every live fitted object, read the saved transform and classifier.
        del models
        _,reloaded=load_frozen_bundle(folder/'models')
        check=predict_frozen(reloaded,Xe,me.age.to_numpy(float),me.sex.to_numpy(float))
        for model,v in pred.items():
            err=float(np.max(np.abs(v['probability_ad']-check[model]['probability_ad'])))
            auc=roc_auc_score(ye,v['probability_ad']);auc2=roc_auc_score(ye,check[model]['probability_ad'])
            dim=int(reloaded[model]['choice']['dimension']); coeflen=len(reloaded[model]['clf_coef'])
            expected=2 if model=='age_sex' else (dim if model.endswith(('_pca','_cpca')) else len(reloaded[model]['transform']['kept_names']))+(2 if model.startswith('age_sex') else 0)
            passed=err<=1e-8 and abs(auc-auc2)<=1e-10 and expected==coeflen
            validations.append({'queue':s.queue,'repeat':rep,'fold':s.fold,'model':model,'n_train':len(tr),'n_test':len(te),'subject_overlap':0,'pair_overlap':0,'max_probability_error':err,'reload_auc':auc2,'reference_auc':auc,'auc_difference':auc2-auc,'dimension':dim,'coef_length':coeflen,'pass':passed})
            for j,sid in enumerate(te):
                all_predictions.append({'queue':s.queue,'repeat':rep,'fold':int(s.fold),'model':model,'subject_id':sid,'pair_id':int(tepair[j]) if tepair is not None else None,'y_true':int(ye[j]),'probability':float(v['probability_ad'][j])})
    p=pd.DataFrame(all_predictions)
    if p.duplicated(['queue','repeat','model','subject_id']).any():raise AssertionError('Subject occurs in multiple outer tests')
    p.to_csv(result/'internal_oof_predictions.csv',index=False)
    val=pd.DataFrame(validations);val.to_csv(result/'model_reload_prediction_validation.csv',index=False)
    pd.DataFrame(inner_ids).to_csv(root/'private/development/inner_splits.csv',index=False)
    rows=[]
    for (q,rep,model),g in p.groupby(['queue','repeat','model']):
        rows.append({'queue':q,'repeat':rep,'model':model,'n':len(g),'roc_auc':roc_auc_score(g.y_true,g.probability),'metric_level':'repeat_combined_oof'})
    metrics=pd.DataFrame(rows);metrics.to_csv(result/'internal_metrics.csv',index=False)
    metrics.groupby(['queue','model']).roc_auc.agg(['mean','std','min','max']).reset_index().to_csv(result/'internal_metric_summary.csv',index=False)
    if not val['pass'].all(): raise RuntimeError('Final fold reload failure')


def freeze(c):
    root=c['_root'];cfg=model_config(c);m,f=development(root);folder=Path(c['paths']['frozen_bundle'])
    (root/'private/results').mkdir(parents=True,exist_ok=True)
    if folder.exists():
        manifest,_=load_frozen_bundle(folder)
        if manifest['source_sha256']!=source_hashes(root) or manifest['versions']!=versions() or manifest['configuration']!=json.loads(json.dumps(asdict(cfg))):
            raise RuntimeError('Frozen training inputs/code/dependencies changed; existing immutable model cannot represent this run. Retain it and create an explicitly versioned analysis.')
        log(root,'freeze','Existing immutable bundle and training fingerprints verified; not overwritten');return
    X=f.to_numpy(float);y=m.group.eq('AD').astype(int).to_numpy();a=m.age.to_numpy(float);sx=m.sex.to_numpy(float)
    selected,table=select_hyperparameters(X,y,m.group.to_numpy(),a,sx,f.columns,cfg,cfg.random_seed)
    table.to_csv(root/'private/development/final_inner_candidate_scores.csv',index=False)
    models=fit_frozen_models(X,y,m.group.to_numpy(),a,sx,f.columns,selected,cfg,cfg.random_seed)
    protocol={'written_utc':pd.Timestamp.now(tz='UTC').isoformat(),'positive_class':'AD','threshold':0.5,'bootstrap':{'n':2000,'seed':20261020,'unit':'subject, stratified by diagnosis; same draws across models'},'paired_comparisons':[['age_adjusted_cpca','age_adjusted_eeg'],['age_adjusted_cpca','age_adjusted_pca'],['age_sex_age_adjusted_cpca','age_sex_age_adjusted_eeg'],['age_sex_age_adjusted_cpca','age_sex_age_adjusted_pca'],['age_sex_age_adjusted_cpca','age_sex'],['age_sex_age_adjusted_eeg','age_sex'],['age_sex_age_adjusted_pca','age_sex']],'age_sensitivity':'within development NC age min/max, report as sensitivity only; primary includes all eligible','permutations':100,'interpretation_groups_sha256':sha(root/'configs/interpretation_groups.json'),'schema_sha256':sha(root/'configs/feature_schema.json'),'mmse_primary':'age_adjusted_cpca cPC1; selection does not use MMSE','mmse_secondary':'all retained components and classifier logit exploratory','mmse_tests':'Spearman AD and pooled; rank residual partial Spearman age/sex (pooled additionally diagnosis); HC3 OLS with same covariates; education sensitivity development only','fdr':'BH over all component/score tests within dataset x statistic family, covering groups and adjustment models','no_variance':'NA; do not compute constant NC MMSE correlation'}
    with (root/'private/evaluation_protocol.json').open('x',encoding='utf-8') as h:json.dump(protocol,h,indent=2)
    manifest=save_frozen_bundle(folder,models,selected,m.index.tolist(),source_hashes(root),cfg,versions())
    pred=predict_frozen(models,X,a,sx); del models
    _,loaded=load_frozen_bundle(folder); check=predict_frozen(loaded,X,a,sx)
    val=[{'model':k,'max_probability_error':float(np.max(abs(v['probability_ad']-check[k]['probability_ad']))),'pass':bool(np.max(abs(v['probability_ad']-check[k]['probability_ad']))<=1e-8)} for k,v in pred.items()]
    pd.DataFrame(val).to_csv(root/'private/results/final_frozen_reload_validation.csv',index=False)
    log(root,'freeze',f"Frozen {len(loaded)} models on {len(m)} subjects",freeze_id=manifest['freeze_id'])
