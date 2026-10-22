from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from .io import development, fingerprint, sha, write_json, versions, table
from .modeling import load_frozen_bundle, predict_frozen, _transform
from .workflow import log

ROIS={'left_frontal':['FP1','F3','F7'],'right_frontal':['FP2','F4','F8'],'left_temporal':['T7'],'right_temporal':['T8'],'central':['C3','C4'],'parietal_posterior':['P3','P4','P7','P8'],'occipital':['O1','O2']}
ROI_OF={c:r for r,cs in ROIS.items() for c in cs}


def feature_dictionary(names):
    rows=[]
    prefixes={'power_log10_':('absolute_power','log10(microvolt^2)'),'relative_log10_':('relative_power','log10(fraction)'),'spectral_entropy_':('spectral_entropy','dimensionless normalized entropy'),'dwpli2_':('connectivity','signed debiased squared wPLI'),'hjorth_complexity_':('Hjorth_complexity','dimensionless Hjorth complexity')}
    for name in names:
        for prefix,(kind,unit) in prefixes.items():
            if name.startswith(prefix):
                rest=name[len(prefix):];break
        else:raise ValueError(f'Unrecognized feature: {name}')
        band=next((b for b in ['gamma30_40','delta','theta','alpha','beta','broad'] if rest.startswith(b+'_')),None)
        if band is None:raise ValueError(f'Unrecognized band: {name}')
        channels=rest[len(band)+1:].split('__');c1=channels[0];c2=channels[1] if len(channels)==2 else ''
        if c1 not in ROI_OF or (c2 and c2 not in ROI_OF):raise ValueError(f'Unknown electrode: {name}')
        r1=ROI_OF[c1];r2=ROI_OF[c2] if c2 else ''
        roi_pair='--'.join(sorted([r1,r2])) if c2 else ''
        rows.append({'feature_name':name,'feature_type':kind,'band':band,'channel_1':c1,'channel_2':c2,'roi_1':r1,'roi_2':r2,'roi_pair':roi_pair,'edge_key':'--'.join(sorted(channels)) if c2 else '', 'unit':unit,'node_or_edge':'edge' if c2 else 'node'})
    return pd.DataFrame(rows)


def groups_for_permutation(dictionary):
    groups={}
    spectral=dictionary.feature_type.isin(['absolute_power','relative_power','spectral_entropy'])
    groups['indicator:spectral_power_and_entropy']=np.flatnonzero(spectral)
    for kind in dictionary.feature_type.unique():groups['indicator:'+kind]=np.flatnonzero(dictionary.feature_type.eq(kind))
    for band in dictionary.band.unique():groups['band:'+band]=np.flatnonzero(dictionary.band.eq(band))
    for roi in ROIS:groups['node_roi:'+roi]=np.flatnonzero(dictionary.node_or_edge.eq('node')&dictionary.roi_1.eq(roi))
    for rp in sorted(dictionary.loc[dictionary.node_or_edge.eq('edge'),'roi_pair'].unique()):groups['connection_roi:'+rp]=np.flatnonzero(dictionary.roi_pair.eq(rp))
    return {k:v for k,v in groups.items() if len(v)}


def auc_rows(y,score):
    """Exact rank AUC, including average ranks for ties, for rows of predictions."""
    y=np.asarray(y,int);score=np.atleast_2d(score);r=rankdata(score,axis=1,method='average');npos=int(y.sum());nneg=len(y)-npos
    return (r[:,y==1].sum(axis=1)-npos*(npos+1)/2)/(npos*nneg)


def residual_features(pack,X,age,sex):
    tr=pack['transform'];Xi=np.where(np.isfinite(X),X,tr['median'])
    if tr['robust_clip']:Xi=np.clip(Xi,tr['qlo'],tr['qhi'])
    return Xi[:,tr['keep']]-np.column_stack([np.ones(len(X)),age,sex])@tr['age_beta']


def backproject(pack,model):
    tr=pack['transform'];demo=model.startswith('age_sex')
    beta=pack['clf_coef'][2:] if demo else pack['clf_coef']
    if model.endswith('_cpca'):v=pack['cpca_components'];b=v@beta;offset=0.
    elif model.endswith('_pca'):v=pack['pca_components'];b=v@beta;offset=-float(pack['pca_mean']@b)
    else:b=beta;offset=0.
    w=tr['block_weights']/tr['scale']*b
    intercept=pack['clf_intercept']-float(tr['center']@w)+offset
    demo_original=np.zeros(2)
    if demo:
        demo_original=pack['clf_coef'][:2]/pack['demo_transform']['scale']
        intercept-=float(pack['demo_transform']['center']@demo_original)
    return w,intercept,demo_original,b*tr['block_weights']


def explain_fitted(models,X,age,sex,names,reference_scale,identity):
    predictions=predict_frozen(models,X,age,sex);rows=[];validation=[]
    for name,pack in models.items():
        if name=='age_sex':continue
        tr=pack['transform'];R=residual_features(pack,X,age,sex);w,intercept,demow,wstd=backproject(pack,name)
        decision=R@w+np.column_stack([age,sex])@demow+intercept
        err=float(np.max(abs(decision-predictions[name]['decision'])))
        if err>1e-8:raise AssertionError(f'Backprojection logit mismatch {identity} {name}: {err}')
        g=predictions[name]['eeg_score'];gc=g-g.mean();var=float(np.var(g,ddof=1))
        pattern=(R-R.mean(axis=0)).T@gc/(len(R)-1)/var if var>1e-20 else np.full(R.shape[1],np.nan)
        keep=tr['keep'];orig=np.zeros(len(names));std=np.zeros(len(names));haufe=np.zeros(len(names));haufe_std=np.zeros(len(names))
        orig[keep]=w;std[keep]=wstd;haufe[keep]=pattern;haufe_std[keep]=pattern/tr['scale']
        validation.append({**identity,'model':name,'max_logit_error':err,'pass':err<=1e-8,'demographic_age_coefficient':demow[0],'demographic_male_coefficient':demow[1],'intercept_residual_coordinate':intercept})
        for j,feature in enumerate(names):
            rows.append({**identity,'model':name,'feature_name':feature,'retained':bool(keep[j]),'weight_original_residual':orig[j],'weight_fold_standardized':std[j],'weight_common_reference':orig[j]*reference_scale[j],'haufe_original_residual':haufe[j],'haufe_fold_standardized':haufe_std[j],'haufe_common_reference':haufe[j]/reference_scale[j]})
    return rows,validation


def permutation_predictions(models,X,age,sex,y,dictionary,nperm,seed):
    groups=groups_for_permutation(dictionary);baseline=predict_frozen(models,X,age,sex);rng=np.random.default_rng(seed)
    shared={group:np.array([rng.permutation(len(X)) for _ in range(nperm)]) for group in groups}
    result={};rows=[]
    for name,pack in models.items():
        if name=='age_sex':continue
        R=residual_features(pack,X,age,sex);w,_,_,_=backproject(pack,name);retained_indices=np.flatnonzero(pack['transform']['keep'])
        b=baseline[name]['decision'];auc=float(auc_rows(y,b)[0]);modelout={}
        for group,allcols in groups.items():
            cols=np.flatnonzero(np.isin(retained_indices,allcols))
            perm=shared[group];block=R[:,cols]
            change=np.einsum('pnc,c->pn',block[perm]-block[None,:,:],w[cols],optimize=True)
            logits=b[None,:]+change;probs=expit(logits);modelout[group]=probs
            importance=auc-auc_rows(y,probs)
            for p,val in enumerate(importance):rows.append({'model':name,'group':group,'permutation':p,'baseline_auc':auc,'permuted_auc':auc-val,'importance':val,'n_features':len(allcols),'n_retained_features':len(cols),'n_subjects':len(y),'seed':seed})
        result[name]=modelout
    return result,rows


def summarize_regions(explanations,dictionary,out):
    e=explanations.merge(dictionary,on='feature_name',validate='many_to_one')
    # Each summary is first calculated within an actual fit, then over fits.
    fitkeys=['cohort','queue','repeat','fold','model','feature_type','band']
    valuecols=['weight_common_reference','haufe_common_reference']
    for kind,groupcols,filename in [('node',['roi_1'],'regional_node_pattern_summary.csv'),('edge',['roi_pair'],'connection_pattern_summary.csv')]:
        a=e[e.node_or_edge.eq(kind)].copy();rows=[]
        for keys,g in a.groupby(fitkeys+groupcols,dropna=False):
            row=dict(zip(fitkeys+groupcols,keys));row['unique_feature_count']=g.feature_name.nunique();row['unique_edge_count']=g.edge_key.nunique() if kind=='edge' else 0
            for v in valuecols:row[v+'_mean']=g[v].mean();row[v+'_mean_abs']=g[v].abs().mean()
            rows.append(row)
        perfit=pd.DataFrame(rows);perfit.to_csv(out/filename.replace('.csv','_per_fit.csv'),index=False)
        keys=['cohort','queue','model','feature_type','band']+groupcols;summary=[]
        for kk,g in perfit.groupby(keys):
            row=dict(zip(keys,kk));row.update({'n_fits':len(g),'unique_feature_count':int(g.unique_feature_count.max()),'unique_edge_count':int(g.unique_edge_count.max())})
            for v in valuecols:
                q=g[v+'_mean'];row[v+'_mean']=q.mean();row[v+'_mean_abs']=g[v+'_mean_abs'].mean();row[v+'_between_fit_sd']=q.std();row[v+'_sign_consistency']=max(float((q>0).mean()),float((q<0).mean()))
            summary.append(row)
        pd.DataFrame(summary).to_csv(out/filename,index=False)
    # Feature signs, ranks and fixed top-20 display frequency; no significance claim.
    e['abs_haufe']=e.haufe_common_reference.abs()
    e['rank_haufe']=e.groupby(['cohort','queue','repeat','fold','model']).abs_haufe.rank(ascending=False,method='min')
    rows=[]
    for keys,g in e.groupby(['cohort','queue','model','feature_name']):
        rows.append(dict(zip(['cohort','queue','model','feature_name'],keys))|{'mean_abs_haufe':g.abs_haufe.mean(),'mean_haufe':g.haufe_common_reference.mean(),'mean_rank':g.rank_haufe.mean(),'top20_frequency':g.rank_haufe.le(20).mean(),'sign_consistency':max(g.haufe_common_reference.gt(0).mean(),g.haufe_common_reference.lt(0).mean()),'n_fits':len(g)})
    pd.DataFrame(rows).to_csv(out/'feature_pattern_stability.csv',index=False)


def run_interpretation(c, *, external_only=False):
    root=c['_root'];out=root/'private/interpretability';out.mkdir(exist_ok=True)
    mf,frozen=load_frozen_bundle(Path(c['paths']['frozen_bundle']));names=mf['feature_order'];dictionary=feature_dictionary(names);dictionary.to_csv(out/'feature_electrode_region_dictionary.csv',index=False)
    reference=np.ones(len(names));tr=frozen['age_adjusted_eeg']['transform'];reference[tr['keep']]=tr['scale']
    write_json(out/'coordinate_definition.json',{'reference':'Residual-feature standard deviations from frozen full-development NC age/sex adjustment; same vector for every fit.','removed_feature_rule':'Zero coefficient/pattern in fixed original schema, retained flag false.','reference_scale':reference.tolist(),'weight':'coefficient per common reference SD','haufe':'Cov(residual/common reference SD, EEG-only linear score)/Var(EEG-only score); fitted training subjects only','sign_consistency':'max(fraction >0, fraction <0) across actual fits; zero counts in denominator','edge_count':'unique unordered electrode pairs within feature band, never sum over repeated fits','permutation_sd':'Monte Carlo spread; NOT a subject sampling CI','node_regions':ROIS,'complexity':'Hjorth temporal complexity, not spatial complexity'})
    dev,df=development(root);allrows=[];allchecks=[];permutations=[];repeatrows=[]
    if external_only:
        readiness=root/'private/results/external_readiness.json'
        if not readiness.exists() or not json.loads(readiness.read_text())['ready']:raise RuntimeError('External reference confirmation required')
        external=table(root/'private/external/subjects.csv').query('included == True')
        ef=table(root/'private/external/features.csv').loc[external.index,names]
        _,rows=permutation_predictions(frozen,ef.to_numpy(float),external.age.to_numpy(float),external.sex.to_numpy(float),external.group.eq('AD').astype(int).to_numpy(),dictionary,100,20261020)
        pd.DataFrame(rows).to_csv(out/'external_grouped_permutation_importance.csv',index=False)
        pp=pd.DataFrame(rows);pp.groupby(['model','group']).importance.agg(['mean','std','min','max']).reset_index().rename(columns={'mean':'mean_auc_drop','std':'permutation_spread_not_subject_CI'}).to_csv(out/'external_grouped_permutation_importance_summary.csv',index=False)
        from .cognition import cognition
        if c.get('analysis',{}).get('mmse',False):cognition(c)
        write_json(out/'external_explanation_fingerprint.json',{'code_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('cognition.py')]},'versions':versions(),'frozen_manifest_sha256':sha(Path(c['paths']['frozen_bundle'])/'freeze_manifest.json'),'feature_matrix_sha256':sha(root/'private/external/features.csv') if (root/'private/external/features.csv').exists() else None,'subject_metadata_sha256':sha(root/'private/external/subjects.csv') if (root/'private/external/subjects.csv').exists() else None,'group_configuration_sha256':sha(root/'configs/interpretation_groups.json'),'permutations':100,'seed':20261020,'fit_called':False})
        log(root,'interpret','External 100 group permutations and frozen-score cognition completed; no re-fitting')
        return
    rr,v=explain_fitted(frozen,df.to_numpy(float),dev.age.to_numpy(float),dev.sex.to_numpy(float),names,reference,{'cohort':'development_frozen_fit','queue':'full','repeat':-1,'fold':-1})
    allrows.extend(rr);allchecks.extend(v)
    # Frozen loadings describe components; these are not predictive contributions.
    loads=[]
    for name,pack in frozen.items():
        if not name.endswith('_cpca'):continue
        for j in range(pack['cpca_components'].shape[1]):
            for k,feat in enumerate(pack['transform']['kept_names']):loads.append({'model':name,'component':j+1,'feature_name':feat,'loading_model_coordinate':pack['cpca_components'][k,j],'coefficient_common_reference':reference[np.flatnonzero(pack['transform']['keep'])[k]]*pack['transform']['block_weights'][k]/pack['transform']['scale'][k]*pack['cpca_components'][k,j],'sign_rule':'largest absolute training loading positive; no MMSE/test label use'})
    pd.DataFrame(loads).merge(dictionary,on='feature_name').to_csv(out/'frozen_component_loadings.csv',index=False)
    splits=pd.read_csv(root/'private/development/outer_splits.csv')
    for queue in splits.queue.unique():
        for rep in sorted(splits.loc[splits.queue.eq(queue),'repeat'].unique()):
            pooled={};pool_ids=[];pool_y=[]
            for fold in sorted(splits.loc[splits.queue.eq(queue)&splits['repeat'].eq(rep),'fold'].unique()):
                folder=root/'private/fold_models'/queue/f'repeat_{rep:02d}_fold_{fold:02d}'
                ident=json.loads((folder/'identity.json').read_text());_,models=load_frozen_bundle(folder/'models')
                train=ident['train_subject_ids'];test=ident['test_subject_ids'];mt=dev.loc[train];me=dev.loc[test]
                expected=splits.loc[splits.queue.eq(queue)&splits['repeat'].eq(rep)&splits.fold.eq(fold)].iloc[0]
                if train!=json.loads(expected.train_subject_ids) or test!=json.loads(expected.test_subject_ids) or set(train)&set(test):
                    raise ValueError('Explanation identity differs from actual outer split')
                rows,checks=explain_fitted(models,df.loc[train].to_numpy(float),mt.age.to_numpy(float),mt.sex.to_numpy(float),names,reference,{'cohort':'development_outer_training','queue':queue,'repeat':rep,'fold':fold})
                allrows.extend(rows);allchecks.extend(checks)
                y=me.group.eq('AD').astype(int).to_numpy();seed=20261020+rep*100+fold
                perm,rows=permutation_predictions(models,df.loc[test].to_numpy(float),me.age.to_numpy(float),me.sex.to_numpy(float),y,dictionary,50,seed)
                for row in rows:permutations.append({'cohort':'development_oof','queue':queue,'repeat':rep,'fold':fold,**row})
                baseline=predict_frozen(models,df.loc[test].to_numpy(float),me.age.to_numpy(float),me.sex.to_numpy(float))
                oof=pd.read_csv(root/'private/results/internal_oof_predictions.csv',dtype={'subject_id':str})
                for model,value in baseline.items():
                    original=oof.loc[oof.queue.eq(queue)&oof['repeat'].eq(rep)&oof.fold.eq(fold)&oof.model.eq(model)].set_index('subject_id').loc[test]
                    if not np.array_equal(original.y_true.to_numpy(int),y) or np.max(abs(original.probability.to_numpy()-value['probability_ad']))>1e-8:
                        raise ValueError('Permutation baseline differs from saved OOF predictions')
                for model,groups in perm.items():
                    for group,probs in groups.items():
                        key=(model,group);pooled.setdefault(key,{'p':[],'b':[]});pooled[key]['p'].append(probs);pooled[key]['b'].append(baseline[model]['probability_ad'])
                pool_y.extend(y);pool_ids.extend(test)
            if len(pool_ids)!=len(set(pool_ids)):raise AssertionError('Repeat OOF subject duplication')
            for (model,group),v in pooled.items():
                probs=np.concatenate(v['p'],axis=1);base=np.concatenate(v['b']);ba=float(auc_rows(pool_y,base)[0]);pa=auc_rows(pool_y,probs)
                for k,a in enumerate(pa):repeatrows.append({'cohort':'development_oof','queue':queue,'repeat':rep,'model':model,'group':group,'permutation':k,'baseline_auc':ba,'permuted_auc':a,'importance':ba-a,'n_subjects':len(pool_y),'aggregation':'five test folds concatenated by unique subject_id before AUC'})
            log(root,'interpret',f'Internal permutation and training patterns {queue} repeat {rep}')
    readiness=root/'private/results/external_readiness.json'
    external_ready=readiness.exists() and json.loads(readiness.read_text())['ready']
    if external_ready:
        external=table(root/'private/external/subjects.csv').query('included == True')
        ef=table(root/'private/external/features.csv').loc[external.index,names]
        _,rows=permutation_predictions(frozen,ef.to_numpy(float),external.age.to_numpy(float),external.sex.to_numpy(float),external.group.eq('AD').astype(int).to_numpy(),dictionary,100,20261020)
        pd.DataFrame(rows).to_csv(out/'external_grouped_permutation_importance.csv',index=False)
    exp=pd.DataFrame(allrows);exp.to_csv(out/'fold_feature_explanations.csv',index=False)
    exp[[x for x in exp.columns if not x.startswith('weight_')]].to_csv(out/'haufe_patterns.csv',index=False)
    pd.DataFrame(allchecks).to_csv(out/'backprojection_logit_validation.csv',index=False)
    pd.DataFrame(permutations).to_csv(out/'grouped_permutation_importance.csv',index=False)
    pd.DataFrame(repeatrows).to_csv(out/'grouped_permutation_importance_repeat.csv',index=False)
    summarize_regions(exp,dictionary,out)
    for filename in ['external_grouped_permutation_importance.csv','grouped_permutation_importance_repeat.csv']:
        if not (out/filename).exists():continue
        pp=pd.read_csv(out/filename);keys=[k for k in ['cohort','queue','model','group'] if k in pp.columns]
        pp.groupby(keys).importance.agg(['mean','std','min','max']).reset_index().rename(columns={'mean':'mean_auc_drop','std':'permutation_and_repeat_spread_not_subject_CI'}).to_csv(out/filename.replace('.csv','_summary.csv'),index=False)
    from .cognition import cognition
    if c.get('analysis',{}).get('mmse',False):cognition(c)
    write_json(out/'explanation_fingerprint.json',{'interpretation_code_sha256':sha(Path(__file__)),'model_code_sha256':sha(Path(__file__).with_name('modeling.py')),'versions':versions(),'development_features_sha256':sha(root/'private/development/features.csv'),'development_subjects_sha256':sha(root/'private/development/subjects.csv'),'matching_sha256':sha(root/'private/development/matched_pairs.csv'),'frozen_manifest_sha256':sha(Path(c['paths']['frozen_bundle'])/'freeze_manifest.json'),'input_features_sha256':sha(root/'private/external/features.csv') if (root/'private/external/features.csv').exists() else None,'input_subjects_sha256':sha(root/'private/external/subjects.csv') if (root/'private/external/subjects.csv').exists() else None,'configuration_sha256':sha(root/'configs/interpretation_groups.json'),'identity_splits_sha256':sha(root/'private/development/outer_splits.csv'),'results':{p.name:sha(p) for p in out.glob('*.csv')}})
