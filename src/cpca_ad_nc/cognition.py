from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import json
from scipy.stats import rankdata, spearmanr, t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from .io import development,table
from .modeling import load_frozen_bundle, predict_frozen, _transform


def bootstrap_spearman(x,y,n_boot=2000,seed=20261020,strata=None):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<4 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan,np.nan
    rng=np.random.default_rng(seed)
    if strata is None:draws=rng.integers(0,len(x),size=(n_boot,len(x)))
    else:
        blocks=[np.flatnonzero(np.asarray(strata)==v) for v in np.unique(strata)]
        draws=np.concatenate([rng.choice(b,size=(n_boot,len(b)),replace=True) for b in blocks],axis=1)
    a=rankdata(x[draws],axis=1);b=rankdata(y[draws],axis=1);a-=a.mean(axis=1,keepdims=True);b-=b.mean(axis=1,keepdims=True)
    den=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1));r=np.divide((a*b).sum(axis=1),den,out=np.full(len(den),np.nan),where=den>0)
    return tuple(np.nanquantile(r,[.025,.975]))


def partial_spearman(x,y,cov):
    A=sm.add_constant(np.column_stack([rankdata(cov[:,j]) for j in range(cov.shape[1])]),has_constant='add')
    rx=rankdata(x);ry=rankdata(y);rx-=A@np.linalg.lstsq(A,rx,rcond=None)[0];ry-=A@np.linalg.lstsq(A,ry,rcond=None)[0]
    if np.std(rx)<1e-12 or np.std(ry)<1e-12:return np.nan,np.nan
    rho=float(np.corrcoef(rx,ry)[0,1]);df=len(x)-np.linalg.matrix_rank(A)-1
    p=2*t.sf(abs(rho)*np.sqrt(df/max(1-rho*rho,1e-15)),df) if df>0 else np.nan
    return rho,float(p)


def fdr(rows,family_keys):
    d=pd.DataFrame(rows);d['q']=np.nan;d['family_tests']=0
    for _,idx in d.groupby(family_keys,dropna=False).groups.items():
        ii=d.loc[idx].index[d.loc[idx,'p'].notna()];d.loc[idx,'family_tests']=len(ii)
        if len(ii):d.loc[ii,'q']=multipletests(d.loc[ii,'p'],method='fdr_bh')[1]
    return d


def cognition(c):
    root=c['_root'];out=root/'private/interpretability';_,models=load_frozen_bundle(Path(c['paths']['frozen_bundle']))
    dev,features=development(root)
    rows=[];score_rows=[];differences=[];feature_tests=[]
    datasets=[('development_exploratory',dev,features)]
    readiness=root/'private/results/external_readiness.json'
    if readiness.exists() and json.loads(readiness.read_text())['ready']:
        ext=table(root/'private/external/subjects.csv').query('included == True')
        ef=table(root/'private/external/features.csv').loc[ext.index]
        datasets.append(('external_frozen',ext,ef))
    for cohort,m,f in datasets:
        age=m.age.to_numpy(float);sex=m.sex.to_numpy(float);group=m.group.eq('AD').astype(int).to_numpy();X=f.to_numpy(float)
        cp=models['age_adjusted_cpca'];components=_transform(X,age,sex,cp['transform'])@cp['cpca_components']
        predictions=predict_frozen(models,X,age,sex)
        scores={f'cPC{k+1}':components[:,k] for k in range(components.shape[1])}
        scores.update({'risk_logit:'+key:v['decision'] for key,v in predictions.items()})
        for name,score in scores.items():
            for j,sid in enumerate(m.index):score_rows.append({'cohort':cohort,'subject_id':sid,'group':m.iloc[j]['group'],'age':age[j],'sex':sex[j],'mmse':m.iloc[j].mmse,'score_name':name,'score':score[j],'primary':name=='cPC1'})
            D=sm.add_constant(np.column_stack([group,age,sex]),has_constant='add')
            fit=sm.OLS(score,D).fit(cov_type='HC3')
            differences.append({'cohort':cohort,'score_name':name,'n':len(m),'ad_minus_nc_age_sex_adjusted':fit.params[1],'ci_lower':fit.conf_int()[1,0],'ci_upper':fit.conf_int()[1,1],'p':fit.pvalues[1],'score_direction':'frozen, never flipped using external effects'})
            for scope in ['AD_only','pooled_AD_NC','NC_only']:
                valid=np.isfinite(m.mmse.to_numpy(float)) & (group==1 if scope=='AD_only' else group==0 if scope=='NC_only' else np.ones(len(m),bool))
                x=score[valid];y=m.mmse.to_numpy(float)[valid];gg=group[valid];n=len(y)
                base={'cohort':cohort,'scope':scope,'score_name':name,'primary_component':name=='cPC1','n':n,'rho':np.nan,'coefficient_mmse_per_sd':np.nan,'ci_lower':np.nan,'ci_upper':np.nan,'p':np.nan,'status':'ok'}
                if n<8 or np.ptp(y)==0 or np.ptp(x)==0:
                    rows.append({**base,'statistic':'Spearman','adjustment':'none','status':'NA: MMSE/score constant or insufficient observations'})
                    continue
                rho,p=spearmanr(x,y);lo,hi=bootstrap_spearman(x,y,strata=gg if scope=='pooled_AD_NC' else None)
                rows.append({**base,'statistic':'Spearman','adjustment':'none','rho':rho,'p':p,'ci_lower':lo,'ci_upper':hi})
                cov=np.column_stack([age[valid],sex[valid]]+([gg] if scope=='pooled_AD_NC' else []))
                adjustment='age+sex'+('+diagnosis' if scope=='pooled_AD_NC' else '')
                if scope=='pooled_AD_NC':
                    r0,p0=partial_spearman(x,y,cov[:,:2])
                    rows.append({**base,'statistic':'partial_Spearman','adjustment':'age+sex','rho':r0,'p':p0})
                    xs=(x-x.mean())/x.std(ddof=1);A0=sm.add_constant(np.column_stack([xs,cov[:,:2]]),has_constant='add');fit0=sm.OLS(y,A0).fit(cov_type='HC3');ci0=fit0.conf_int()[1]
                    rows.append({**base,'statistic':'OLS_HC3','adjustment':'age+sex','coefficient_mmse_per_sd':fit0.params[1],'p':fit0.pvalues[1],'ci_lower':ci0[0],'ci_upper':ci0[1]})
                r,p=partial_spearman(x,y,cov)
                rows.append({**base,'statistic':'partial_Spearman','adjustment':adjustment,'rho':r,'p':p})
                for extra in [False,True] if cohort=='development_exploratory' else [False]:
                    mask=np.ones(n,bool);C=cov.copy();adj=adjustment
                    if extra:
                        edu=m.education_years.to_numpy(float)[valid];mask=np.isfinite(edu);C=np.column_stack([C,edu]);adj+='+education_sensitivity'
                    if mask.sum()<C.shape[1]+5:continue
                    xx=x[mask];yy=y[mask];cc=C[mask];xx=(xx-xx.mean())/xx.std(ddof=1)
                    A=sm.add_constant(np.column_stack([xx,cc]),has_constant='add')
                    if np.linalg.matrix_rank(A)<A.shape[1]:continue
                    fit=sm.OLS(yy,A).fit(cov_type='HC3');ci=fit.conf_int()[1]
                    rows.append({**base,'n':len(yy),'statistic':'OLS_HC3','adjustment':adj,'coefficient_mmse_per_sd':fit.params[1],'p':fit.pvalues[1],'ci_lower':ci[0],'ci_upper':ci[1]})
                    if extra:
                        r,p=partial_spearman(xx,yy,cc);rows.append({**base,'n':len(yy),'statistic':'partial_Spearman','adjustment':adj,'rho':r,'p':p})
        # Direction of actual numerical EEG metrics, separate from model weights.
        for feature in f.columns:
            vals=f[feature].to_numpy(float);valid=np.isfinite(vals);A=sm.add_constant(np.column_stack([group[valid],age[valid],sex[valid]]),has_constant='add')
            fit=sm.OLS(vals[valid],A).fit(cov_type='HC3')
            feature_tests.append({'cohort':cohort,'feature_name':feature,'n':int(valid.sum()),'ad_minus_nc_age_sex_adjusted':fit.params[1],'p':fit.pvalues[1]})
    results=fdr(rows,['cohort','statistic']);results.to_csv(out/'mmse_associations_all_planned.csv',index=False)
    pd.DataFrame(score_rows).to_csv(out/'frozen_component_and_risk_scores.csv',index=False)
    fdr(differences,['cohort']).to_csv(out/'component_and_model_group_directions.csv',index=False)
    dictionary=pd.read_csv(out/'feature_electrode_region_dictionary.csv')
    tests=pd.DataFrame(feature_tests).merge(dictionary,on='feature_name',validate='many_to_one')
    fdr(tests.to_dict('records'),['cohort','feature_type']).to_csv(out/'actual_feature_group_directions_fdr.csv',index=False)
