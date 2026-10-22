from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tomllib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def fingerprint(*values):
    h=hashlib.sha256()
    for v in values:
        if isinstance(v,np.ndarray):
            h.update(str(v.shape).encode());h.update(str(v.dtype).encode());h.update(v.tobytes())
        else:h.update(json.dumps(v,sort_keys=True,default=str).encode())
    return h.hexdigest()


def versions():
    return {k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','h5py','statsmodels','threadpoolctl']}


def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def config(path):
    path=Path(path).resolve()
    with path.open('rb') as f:c=tomllib.load(f)
    base=path.parent
    c['_config_path']=path
    c.setdefault('paths',{})
    for k,v in c['paths'].items():
        if v:
            p=Path(v).expanduser();c['paths'][k]=str(p if p.is_absolute() else (base/p).resolve())
    root=Path(c['paths'].get('workspace',base/'../private_run')).resolve()
    c['_root']=root
    c['paths'].setdefault('frozen_bundle',str(root/'private/frozen'))
    for d in ['configs','private/results','runs']:(root/d).mkdir(parents=True,exist_ok=True)
    assets=Path(__file__).parent/'assets'
    for name in ['feature_schema.json','interpretation_groups.json']:
        src=assets/name;dst=root/'configs'/name
        if dst.exists() and sha(src)!=sha(dst):
            raise ValueError('Workspace schema/group definition changed; use a new workspace')
        if not dst.exists():shutil.copyfile(src,dst)
    return c


def table(path):
    p=Path(path)
    d=pd.read_csv(p,sep='\t' if p.suffix.lower()=='.tsv' else ',',dtype={'subject_id':'string'})
    if 'subject_id' not in d:raise ValueError('Input table requires subject_id')
    ids=d.subject_id
    if ids.isna().any() or ids.str.strip().eq('').any() or ids.duplicated().any():raise ValueError('Missing/duplicate subject_id')
    if not ids.eq(ids.str.strip()).all():raise ValueError('Leading/trailing whitespace in subject_id')
    return d.set_index('subject_id',verify_integrity=True)


def schema(c):
    return json.loads((c['_root']/'configs/feature_schema.json').read_text(encoding='utf-8'))['feature_order']


def check_features(f,names):
    if f.columns.tolist()!=names:raise ValueError('Feature names/order must exactly match the fixed 752-column schema')
    return f.apply(pd.to_numeric,errors='raise')


def development(root):
    p=Path(root)/'private/development'
    m=table(p/'subjects.csv');f=table(p/'features.csv')
    if set(m.index)!=set(f.index):raise ValueError('Development metadata/feature ID mismatch')
    return m,f.loc[m.index]


def _read_inputs(c,cohort,require_labels):
    paths=c['paths']
    feature_path=paths.get(cohort+'_features')
    if not feature_path:
        feature_path=c['_root']/'private'/cohort/'extracted_features.csv'
        subject_path=c['_root']/'private'/cohort/'extracted_subjects.csv'
    else:subject_path=paths[cohort+'_subjects']
    m=table(subject_path);f=check_features(table(feature_path),schema(c))
    if set(m.index)!=set(f.index):raise ValueError(f'{cohort} metadata/feature identity mismatch')
    if not {'age','sex'}<=set(m):raise ValueError('Metadata requires age and sex (female=0, male=1)')
    for col in ['age','sex','mmse','education_years']:
        if col not in m:m[col]=np.nan
        m[col]=pd.to_numeric(m[col],errors='raise')
    if 'group' not in m:m['group']=''
    reasons=[]
    for sid,row in m.iterrows():
        why=[]
        if require_labels and row['group'] not in ['AD','NC']:why.append('diagnosis_not_AD_NC')
        if not np.isfinite(row.age):why.append('missing_age')
        if not np.isfinite(row.sex) or row.sex not in [0,1]:why.append('missing_or_invalid_sex')
        if not np.isfinite(f.loc[sid].to_numpy(float)).any():why.append('no_valid_EEG_features')
        if 'eeg_valid' in m and str(row.eeg_valid).lower() not in ['true','1']:why.append('EEG_QC_failed')
        reasons.append(';'.join(why))
    m['exclusion_reason']=reasons;m['included']=m.exclusion_reason.eq('')
    return m,f.loc[m.index]


def _match(m):
    """Study's same-sex optimal one-to-one assignment, fixed +/-3 years."""
    a=m[m.group.eq('NC')];b=m[m.group.eq('AD')];rows=[]
    for sex in sorted(set(a.sex)&set(b.sex)):
        aa=a[a.sex.eq(sex)];bb=b[b.sex.eq(sex)]
        diff=np.abs(aa.age.to_numpy()[:,None]-bb.age.to_numpy()[None,:]);na,nb=diff.shape
        cost=np.full((na,nb+na),1000.);cost[:,:nb]=np.where(diff<=3,diff,1e6)
        ii,jj=linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            if j<nb and diff[i,j]<=3:
                rows.append({'a_subject_id':aa.index[i],'b_subject_id':bb.index[j],'age_diff':diff[i,j],'sex_code':sex})
    order={sid:i for i,sid in enumerate(m.index)}
    rows.sort(key=lambda x:(order[x['a_subject_id']],order[x['b_subject_id']]))
    return pd.DataFrame([{'pair_id':i,**r} for i,r in enumerate(rows)],columns=['pair_id','a_subject_id','b_subject_id','age_diff','sex_code'])


def validate_pairs(m,pairs):
    ids=pairs.a_subject_id.tolist()+pairs.b_subject_id.tolist()
    if len(ids)!=len(set(ids)) or pairs.pair_id.duplicated().any():raise ValueError('Duplicate matched identity/pair')
    if not set(ids)<=set(m.index):raise ValueError('Matched identity absent from eligible cohort')
    for r in pairs.itertuples():
        a=m.loc[r.a_subject_id];b=m.loc[r.b_subject_id]
        if {a['group'],b['group']}!={'AD','NC'} or a.sex!=b.sex or abs(a.age-b.age)>3:raise ValueError('Invalid same-sex +/-3y pair')


def _outer_splits(c,m,pairs):
    from .workflow import model_config
    cfg=model_config(c);rows=[]
    pairmap={str(s):int(r.pair_id) for r in pairs.itertuples() for s in [r.a_subject_id,r.b_subject_id]}
    for queue in ['full','matched_3y_NC_AD']:
        sub=m if queue=='full' else m.loc[m.index.isin(pairmap)]
        if not len(sub):continue
        y=sub.group.eq('AD').astype(int).to_numpy();ids=sub.index.to_numpy()
        if queue=='full':units=np.arange(len(sub));strata=y
        else:
            first=pd.DataFrame({'pair':[pairmap[s] for s in ids],'y':y}).drop_duplicates('pair')
            units=first.pair.to_numpy();strata=first.y.to_numpy()
        n=min(cfg.outer_splits,int(pd.Series(strata).value_counts().min()))
        if n<2:raise ValueError('Insufficient subjects/pairs for outer CV')
        for rep in range(cfg.outer_repeats):
            for fold,(tr,te) in enumerate(StratifiedKFold(n_splits=n,shuffle=True,random_state=cfg.random_seed+rep).split(units,strata)):
                if queue!='full':
                    labels=np.array([pairmap[s] for s in ids]);tr=np.flatnonzero(np.isin(labels,units[tr]));te=np.flatnonzero(np.isin(labels,units[te]))
                rows.append({'queue':queue,'repeat':rep,'fold':fold,'train_subject_ids':json.dumps(ids[tr].tolist()),'test_subject_ids':json.dumps(ids[te].tolist())})
    return pd.DataFrame(rows)


def validate_splits(m,pairs,splits):
    required={'queue','repeat','fold','train_subject_ids','test_subject_ids'}
    if not required<=set(splits) or splits.empty:raise ValueError('Missing/empty split schema')
    if splits.duplicated(['queue','repeat','fold']).any():raise ValueError('Duplicate split key')
    pairmap={s:int(r.pair_id) for r in pairs.itertuples() for s in [r.a_subject_id,r.b_subject_id]}
    for (queue,rep),g in splits.groupby(['queue','repeat']):
        if queue not in ['full','matched_3y_NC_AD']:raise ValueError('Unknown queue')
        expected=set(m.index) if queue=='full' else set(pairmap);seen=[]
        for r in g.itertuples():
            tr=json.loads(r.train_subject_ids);te=json.loads(r.test_subject_ids)
            if len(tr)!=len(set(tr)) or len(te)!=len(set(te)) or set(tr)&set(te) or set(tr)|set(te)!=expected:raise ValueError('Outer identity partition invalid')
            if queue!='full' and {pairmap[s] for s in tr}&{pairmap[s] for s in te}:raise ValueError('Outer pair overlap')
            for ids in [tr,te]:
                if set(m.loc[ids,'group'])!={'AD','NC'}:raise ValueError('Each train/test split requires both classes')
            seen+=te
        if len(seen)!=len(set(seen)) or set(seen)!=expected:raise ValueError('Repeat test sets do not cover queue exactly once')


def prepare(c):
    root=c['_root'];m,f=_read_inputs(c,'development',True);out=root/'private/development';out.mkdir(parents=True,exist_ok=True)
    key=fingerprint(sha(Path(__file__)),m.to_dict('split'),f.to_numpy(float),f.columns.tolist(),c.get('model',{}),
                    {k:sha(c['paths'][k]) for k in ['outer_splits','matched_pairs'] if c['paths'].get(k)},versions())
    marker=out/'preparation.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        if previous['key']==key and all(sha(out/n)==v for n,v in previous['output_sha256'].items()):
            print('[prepare] cache hit: inputs/config/code/dependencies and prepared outputs match',flush=True);return
        if (root/'private/fold_models').exists() or Path(c['paths']['frozen_bundle']).exists():raise ValueError('Prepared inputs changed after fitting; retain this run and choose a new workspace')
    m.to_csv(out/'all_subjects.csv');f.to_csv(out/'all_features.csv')
    m=m.loc[m.included];f=f.loc[m.index]
    if set(m.group)!={'AD','NC'}:raise ValueError('Eligible development cohort requires AD and NC')
    m.to_csv(out/'subjects.csv');f.to_csv(out/'features.csv')
    pairs=pd.read_csv(c['paths']['matched_pairs'],dtype={'a_subject_id':str,'b_subject_id':str}) if c['paths'].get('matched_pairs') else _match(m)
    validate_pairs(m,pairs);pairs.to_csv(out/'matched_pairs.csv',index=False)
    splits=pd.read_csv(c['paths']['outer_splits']) if c['paths'].get('outer_splits') else _outer_splits(c,m,pairs)
    validate_splits(m,pairs,splits);splits.to_csv(out/'outer_splits.csv',index=False)
    write_json(marker,{'key':key,'status':'recomputed','eligible':len(m),'pairs':len(pairs),'output_sha256':{n:sha(out/n) for n in ['subjects.csv','features.csv','outer_splits.csv','matched_pairs.csv']}})
    print(f'[prepare] {len(m)} eligible; {len(pairs)} matched pairs; MMSE not an eligibility criterion',flush=True)


def prepare_external(c,require_labels=True):
    root=c['_root'];m,f=_read_inputs(c,'external',require_labels)
    if c.get('features',{}).get('unit_token')!='EEGLAB_microvolt':raise ValueError('External feature units require verified EEGLAB_microvolt token')
    if not m.included.all():raise ValueError('External inputs contain ineligible rows; inspect metadata/QC and supply the prespecified eligible cohort')
    dev=root/'private/development/subjects.csv'
    if dev.exists() and set(table(dev).index)&set(m.index):raise ValueError('Development/external subject_id overlap')
    out=root/'private/external';out.mkdir(parents=True,exist_ok=True)
    m.to_csv(out/'subjects.csv');f.to_csv(out/'features.csv')
    write_json(root/'private/results/external_readiness.json',{'ready':True,'input_mode':'schema features','unit':c.get('features',{}).get('unit_token','EEGLAB_microvolt')})


def extract_features(c,cohort):
    """Read-only audited EEGLAB v7.3/.fdt extractor; no preprocessing."""
    from dataclasses import asdict
    from .features import FeatureConfig,extract_subject_features
    from .workflow import log
    p=c['paths'];m=table(p[cohort+'_subjects'])
    if 'set_path' not in m:raise ValueError('EEG extraction requires set_path in subject table')
    options=c.get('features',{});confirmed=options.get('confirmed_rest_reference',False)
    if options.get('unit_token')!='EEGLAB_microvolt':raise ValueError('Verify stored EEGLAB microvolt units before extraction')
    if confirmed and not options.get('reference_evidence','').strip():raise ValueError('REST metadata override requires documented evidence')
    cfg=FeatureConfig(confirmed_rest_reference=confirmed);root=c['_root'];out=root/'private'/cohort;cache=out/'feature_cache';cache.mkdir(parents=True,exist_ok=True)
    rows=[];qc=[];base=Path(p[cohort+'_subjects']).parent
    for k,(sid,row) in enumerate(m.iterrows()):
        path=Path(row.set_path);path=path if path.is_absolute() else base/path
        key=fingerprint(sha(Path(__file__).with_name('features.py')),asdict(cfg),sha(path),sha(path.with_suffix('.fdt')),versions())
        cp=cache/f'{fingerprint(sid)[:20]}.json';hit=cp.exists() and json.loads(cp.read_text(encoding='utf-8'))['key']==key
        try:
            if hit:
                previous=json.loads(cp.read_text(encoding='utf-8'));feat=previous['features'];meta=previous['metadata']
            else:
                feat,meta=extract_subject_features(path,cfg);write_json(cp,{'key':key,'features':feat,'metadata':meta})
            if set(feat)!=set(schema(c)):raise ValueError('Extracted feature schema mismatch')
            rows.append({'subject_id':sid,**feat});qc.append({'subject_id':sid,'eeg_valid':True,'reason':'','metadata':json.dumps(meta,default=str)})
        except (ValueError,RuntimeError) as e:
            rows.append({'subject_id':sid,**dict.fromkeys(schema(c),np.nan)});qc.append({'subject_id':sid,'eeg_valid':False,'reason':str(e)})
        log(root,'features',f'{cohort} subject {k+1}/{len(m)}',status='hit' if hit else 'recomputed',key=key)
    fp=out/'extracted_features.csv';pd.DataFrame(rows).set_index('subject_id')[schema(c)].to_csv(fp)
    q=pd.DataFrame(qc).set_index('subject_id');q.to_csv(out/'extraction_qc.csv')
    m['eeg_valid']=q.eeg_valid;m.to_csv(out/'extracted_subjects.csv')
    p[cohort+'_features']=str(fp);p[cohort+'_subjects']=str(out/'extracted_subjects.csv')

