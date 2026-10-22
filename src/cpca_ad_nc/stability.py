from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from .io import development
from .modeling import ModelConfig,_fit_transform,_transform,cpca_fit,load_frozen_bundle
from .workflow import model_config,log


def common_loadings(tr,V,names,reference):
    B=np.zeros((len(names),V.shape[1]));B[tr['keep']]=reference[tr['keep'],None]*(tr['block_weights']/tr['scale'])[:,None]*V
    return B


def fit_loadings(X,g,a,s,names,cfg,alpha,dim,reference):
    tr=_fit_transform(X,names,g,a,s,cfg);Z=_transform(X,a,s,tr)
    cp=cpca_fit(Z[g=='AD'],Z[g=='NC'],alpha)
    return common_loadings(tr,cp['components'][:,:dim],names,reference),tr


def run_stability(c):
    root=c['_root'];cfg=model_config(c);m,f=development(root);_,models=load_frozen_bundle(Path(c['paths']['frozen_bundle']))
    pack=models['age_adjusted_cpca'];tr=pack['transform'];alpha=pack['choice']['alpha'];dim=pack['choice']['dimension']
    names=f.columns.tolist();reference=np.ones(len(names));reference[tr['keep']]=tr['scale']
    baseline=common_loadings(tr,pack['cpca_components'],names,reference)
    X=f.to_numpy(float);g=m.group.to_numpy();a=m.age.to_numpy(float);s=m.sex.to_numpy(float)
    rng=np.random.default_rng(cfg.random_seed);boots=[];bloads=[];loo=[]
    for k in range(200):
        idx=np.r_[rng.choice(np.flatnonzero(g=='AD'),sum(g=='AD'),replace=True),rng.choice(np.flatnonzero(g=='NC'),sum(g=='NC'),replace=True)]
        B,tfit=fit_loadings(X[idx],g[idx],a[idx],s[idx],names,cfg,alpha,dim,reference)
        angles=np.rad2deg(subspace_angles(baseline,B));bn=baseline/np.linalg.norm(baseline,axis=0);cn=B/np.linalg.norm(B,axis=0)
        ri,ci=linear_sum_assignment(-np.abs(bn.T@cn));aligned=B[:,ci].copy();sign=np.sign(np.sum(baseline[:,ri]*aligned,axis=0));aligned*=sign
        boots.append({'bootstrap':k,'alpha':alpha,'dimension':dim,'max_principal_angle_deg':float(angles.max()),'mean_principal_angle_deg':float(angles.mean()),'n_features_retained':int(tfit['keep'].sum()),'stability_type':'conditional on fixed alpha/dimension; stratified subject bootstrap; complete preprocessing refitted','component_matching':'normalized common-coordinate cosine, Hungarian; subspace angles primary'})
        bloads.append(aligned)
    for omit,sid in enumerate(m.index):
        idx=np.delete(np.arange(len(m)),omit);B,tfit=fit_loadings(X[idx],g[idx],a[idx],s[idx],names,cfg,alpha,dim,reference)
        angles=np.rad2deg(subspace_angles(baseline,B));cos=abs(float(np.dot(baseline[:,0],B[:,0])/(np.linalg.norm(baseline[:,0])*np.linalg.norm(B[:,0]))))
        loo.append({'subject_id':sid,'group':g[omit],'participated_in_ad_nc_fit':True,'alpha':alpha,'dimension':dim,'max_principal_angle_deg':float(angles.max()),'mean_principal_angle_deg':float(angles.mean()),'abs_normalized_cpc1_cosine_common_reference':cos,'coordinate':'common full-development residual feature SD','mci_participated':False})
    out=root/'private/interpretability';pd.DataFrame(boots).to_csv(out/'conditional_bootstrap_subspace_stability.csv',index=False);pd.DataFrame(loo).to_csv(out/'ad_nc_leave_one_out_influence.csv',index=False)
    A=np.stack(bloads);np.savez_compressed(out/'conditional_bootstrap_aligned_common_loadings.npz',feature_names=np.array(names),baseline=baseline,bootstrap_loadings=A,reference_scale=reference)
    pd.DataFrame({'feature_name':np.repeat(names,dim),'component':np.tile(np.arange(1,dim+1),len(names)),'common_loading':baseline.ravel(),'bootstrap_lower':np.quantile(A,.025,axis=0).ravel(),'bootstrap_upper':np.quantile(A,.975,axis=0).ravel()}).to_csv(out/'conditional_bootstrap_loading_intervals.csv',index=False)
    log(root,'stability',f'200 grouped bootstrap fits and {len(m)} leave-one-out fits completed; fixed configuration, common coordinates')
