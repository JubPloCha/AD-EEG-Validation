import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from threadpoolctl import threadpool_limits
from cpca_ad_nc.features import _continuous_intervals, _make_segments, FeatureConfig, _one_sided_psd, _integrate_band, h5_chan_labels
from cpca_ad_nc.modeling import ModelConfig, fit_frozen_models, predict_frozen, save_frozen_bundle, load_frozen_bundle, _inner_splits
from cpca_ad_nc.external import predict_frame
from cpca_ad_nc.io import fingerprint


class CoreTests(unittest.TestCase):
    def test_known_sinusoidal_power_and_channel_alias_collision(self):
        import h5py
        fs=500.;time=np.arange(500)/fs
        x=3*np.sin(2*np.pi*10*time)
        frequencies,psd=_one_sided_psd(x.reshape(1,1,-1),fs)
        power=_integrate_band(frequencies,psd,8.,13.)
        self.assertAlmostEqual(float(power[0,0]),3**2/2,places=10)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'channels.h5'
            with h5py.File(p,'w') as f:
                refs=[]
                for k,label in enumerate(['T3','T7']):
                    ds=f.create_dataset('label'+str(k),data=np.array([ord(v) for v in label],dtype=np.uint16));refs.append(ds.ref)
                f.create_dataset('chanlocs/labels',data=np.array(refs,dtype=h5py.ref_dtype))
                with self.assertRaisesRegex(ValueError,'Duplicate channel'):h5_chan_labels(f)

    def test_boundary_duration_is_history(self):
        for dur in [100,float('nan')]:
            events=[{'type':'boundary','latency_matlab':100.5,'duration_samples':dur}]
            intervals,_=_continuous_intervals(1000,events)
            self.assertEqual(intervals,[(0,100),(100,1000)])
            self.assertEqual(sum(b-a for a,b in intervals),1000)
        events=[{'type':'boundary','latency_matlab':x} for x in [.5,100.5,100.5,500.5,1000.5,9999]]
        ints,audit=_continuous_intervals(1000,events)
        self.assertEqual(ints,[(0,100),(100,500),(500,1000)])
        self.assertEqual(sum(not e['valid'] for e in audit),1)
        data=np.arange(1000,dtype=float).reshape(1,1000,1)
        segs,windows=_make_segments(data,{'srate':100,'events':events},FeatureConfig(fs_expected=100,effective_window_sec=1))
        self.assertTrue(all(w['start']>=w['interval_left'] and w['stop']<=w['interval_right'] for w in windows))

    def test_correct_principal_angles(self):
        A=np.array([[.5,0],[0,.5],[0,0.]])
        for B in [A,-A,A[:,::-1],A@np.array([[2.,3.],[1.,4.]])]:
            np.testing.assert_allclose(subspace_angles(A,B),0,atol=1e-14)
        self.assertAlmostEqual(float(subspace_angles(np.eye(3)[:,:1],np.eye(3)[:,1:2])[0]),np.pi/2)

    def test_pair_integrity(self):
        pair=np.repeat(np.arange(24),2);y=np.tile([0,1],24)
        for a,b in _inner_splits(y,3,42,pair):
            self.assertFalse(set(pair[a])&set(pair[b]))

    def test_freeze_reload_external_invariance_and_schema(self):
        rng=np.random.default_rng(11);n=48;names=['power_log10_theta_FP1','relative_log10_alpha_FP2','dwpli2_theta_FP1__FP2','hjorth_complexity_broad_FP1']
        X=rng.normal(size=(n,4));g=np.array(['AD','NC']*(n//2));y=(g=='AD').astype(int);age=rng.normal(65,8,n);sex=np.tile([0,0,1,1],n//4)
        from cpca_ad_nc.modeling import MODEL_NAMES
        selected={k:{'dimension':2 if k.endswith(('_pca','_cpca')) else 0,'alpha':.3,'C':1.} for k in MODEL_NAMES}
        cfg=ModelConfig(dim_grid=(1,2))
        with threadpool_limits(limits=1):models=fit_frozen_models(X,y,g,age,sex,names,selected,cfg,42)
        ref=predict_frozen(models,X,age,sex)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'bundle';save_frozen_bundle(folder,models,selected,[f'synthetic:{i}' for i in range(n)],{},cfg,{})
            _,loaded=load_frozen_bundle(folder)
            f=pd.DataFrame(X,index=[f'synthetic:{i}' for i in range(n)],columns=names)
            m=pd.DataFrame({'age':age,'sex':sex,'group':g},index=f.index)
            p=predict_frame(loaded,f,m);rev=predict_frame(loaded,f.iloc[::-1],m)
            for k in models:
                np.testing.assert_allclose(p[k]['probability_ad'],ref[k]['probability_ad'],atol=1e-14)
                np.testing.assert_allclose(p[k]['probability_ad'],rev[k]['probability_ad'][::-1],atol=1e-14)
            with self.assertRaises(ValueError):predict_frame(loaded,f.iloc[:,::-1],m)
            with self.assertRaises(ValueError):predict_frame(loaded,f,m,unit_token='V')
            with self.assertRaises(ValueError):predict_frame(loaded,pd.concat([f,f.iloc[:1]]),m)
            m.iloc[0,m.columns.get_loc('age')]=np.nan
            with self.assertRaises(ValueError):predict_frame(loaded,f,m)
            (folder/'age_sex.json').write_text('{}')
            with self.assertRaises(ValueError):load_frozen_bundle(folder)

    def test_cache_fingerprints(self):
        X=np.arange(12).reshape(4,3)
        base=fingerprint('boundary_code_v1',{'window':1},X,['a','b'])
        self.assertEqual(base,fingerprint('boundary_code_v1',{'window':1},X,['a','b']))
        for change in [fingerprint('boundary_code_v2',{'window':1},X,['a','b']),fingerprint('boundary_code_v1',{'window':2},X,['a','b']),fingerprint('boundary_code_v1',{'window':1},X+1,['a','b']),fingerprint('boundary_code_v1',{'window':1},X,['a','c'])]:
            self.assertNotEqual(base,change)


if __name__=='__main__':unittest.main()
