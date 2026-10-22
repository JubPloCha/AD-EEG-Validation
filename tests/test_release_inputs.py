import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from cpca_ad_nc.io import config,prepare,development,validate_splits,validate_pairs


class ReleaseInputs(unittest.TestCase):
    def test_preparation_preserves_ids_pairs_and_rejects_last_fold_corruption(self):
        from cpca_ad_nc.io import _match,_outer_splits
        n=80;idx=pd.Index([f'SYN_{i:03d}' for i in range(n)],name='subject_id')
        m=pd.DataFrame({'group':np.tile(['NC','AD'],n//2),'age':np.repeat(np.arange(n//2)%20+55,2),'sex':np.repeat(np.arange(n//2)%2,2)},index=idx)
        pairs=_match(m);validate_pairs(m,pairs)
        c={'model':{'outer_splits':2,'outer_repeats':1}}
        splits=_outer_splits(c,m,pairs);validate_splits(m,pairs,splits)
        first=json.loads(splits.iloc[0].test_subject_ids);second=json.loads(splits.iloc[1].test_subject_ids)
        self.assertNotEqual(first,second)
        corrupt=splits.copy();corrupt.loc[0,'test_subject_ids']=corrupt.loc[1,'test_subject_ids']
        with self.assertRaises(ValueError):validate_splits(m,pairs,corrupt)
        bad=pairs.copy();bad.loc[0,'b_subject_id']=bad.loc[1,'b_subject_id']
        with self.assertRaises(ValueError):validate_pairs(m,bad)

    def test_unicode_config_and_exact_feature_order(self):
        from cpca_ad_nc.io import check_features,schema
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'中文 path';base.mkdir()
            p=base/'config.local.toml';p.write_text("[paths]\nworkspace = 'run with spaces'\n",encoding='utf-8')
            c=config(p);names=schema(c)
            self.assertEqual(len(names),752)
            self.assertEqual(c['_root'],base/'run with spaces')
            f=pd.DataFrame(np.zeros((1,752)),columns=names)
            check_features(f,names)
            with self.assertRaises(ValueError):check_features(f.iloc[:,::-1],names)


if __name__=='__main__':unittest.main()
