"""Recompute both selection and expanded scores against canonical raw labels."""
import json
from pathlib import Path
import numpy as np
from .transport_train import ROOT
from .data import Dataset,write_json,sha
from .metrics import evaluate


def audit():
    data=Dataset(device='cpu');receipts=[]
    for completed in sorted(ROOT.glob('*/completion.json')):
        run=completed.parent;config=json.loads((run/'config.json').read_text(encoding='utf-8'))
        assert config['data_sha256']==data.contract['data_sha256']
        folders=[run]+[p.parent for p in run.glob('*/evaluation.npz')]
        for folder in folders:
            if folder==run:rows=[0,3,6,9];hours=4
            else:
                conf=json.loads((folder/'config.json').read_text(encoding='utf-8'));rows=conf['rows'];hours=conf['hours']
                assert conf['data_sha256']==data.contract['data_sha256']
                assert conf['checkpoint_sha256']==sha(run/'best.pt')
            p=np.load(folder/'evaluation.npz');saved=json.loads((folder/'metrics.json').read_text())
            canonical=np.concatenate([data.a['val_y'][r,12:12+hours] for r in rows])
            expected=np.broadcast_to(canonical,p['truth'].shape)
            np.testing.assert_allclose(p['truth'],expected,rtol=1e-6,atol=5e-5,equal_nan=True)
            assert np.isfinite(p['samples']).all();difference=0
            for li in range(3):
                result=evaluate(p['samples'][li],p['truth'][li])
                for key,value in result.items():difference=max(difference,abs(value-saved['layouts'][li][key]))
                if hours>1:
                    ds=np.concatenate([np.diff(p['samples'][li,:,i:i+hours],axis=1) for i in range(0,len(canonical),hours)],1)
                    dy=np.concatenate([np.diff(p['truth'][li,i:i+hours],axis=0) for i in range(0,len(canonical),hours)])
                    inc=evaluate(ds,dy)
                    for key,value in inc.items():difference=max(difference,abs(value-saved['layouts'][li]['increment'][key]))
            if difference>1e-5:raise ValueError('Metric difference exceeds tolerance')
            receipts.append(dict(run=run.name,scope='selection' if folder==run else folder.name,max_difference=difference,
                                 matches_canonical_truth=True,all_predictions_finite=True,valid_truth_count=int(np.isfinite(p['truth']).sum())))
    write_json(ROOT/'artifact_audit.json',dict(status='passed',receipts=receipts));print(f'Audited {len(receipts)} prediction artifacts')


if __name__=='__main__':audit()
