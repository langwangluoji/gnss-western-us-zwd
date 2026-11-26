"""Recompute saved scores and verify artifact population/provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import Dataset,write_json,sha
from .metrics import evaluate
from .train import checkpoint_receipt


def audit(root):
    root=Path(root);receipts=[];population=None
    for path in sorted(root.glob('*/completion.json')):
        run=path.parent;config=json.loads((run/'config.json').read_text(encoding='utf-8'));data=Dataset(config['data'],'cpu')
        assert data.contract['data_sha256']==config['data_sha256']
        p=np.load(run/'evaluation.npz');samples=p['samples'];truth=p['truth']
        if population is None:population=truth
        else:np.testing.assert_equal(truth,population)
        old=json.loads((run/'metrics.json').read_text(encoding='utf-8'));diff=0
        for i in range(3):
            new=evaluate(samples[i],truth[i])
            for key in ('station_macro_rmse_mm','micro_rmse_mm','mae_mm','bias_mm','fair_crps_mm','coverage90','width90_mm'):
                diff=max(diff,abs(new[key]-old['layouts'][i][key]))
        receipt=dict(run=run.name,all_predictions_finite=bool(np.isfinite(samples).all()),metric_max_difference=diff,
                     samples_shape=list(samples.shape),valid_truth_count=int(np.isfinite(truth).sum()),
                     checkpoint_sha256=sha(run/'best.pt'),encoder=checkpoint_receipt(config.get('encoder')))
        if diff>1e-5 or not receipt['all_predictions_finite']:raise ValueError(f'Artifact audit failed: {run.name}')
        write_json(run/'postrun_audit.json',receipt);receipts.append(receipt)
    write_json(root/'audit.json',dict(status='passed',same_truth_population=True,runs=receipts))
    print(json.dumps(dict(runs=len(receipts),max_difference=max(r['metric_max_difference'] for r in receipts))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');audit(p.parse_args().root)
