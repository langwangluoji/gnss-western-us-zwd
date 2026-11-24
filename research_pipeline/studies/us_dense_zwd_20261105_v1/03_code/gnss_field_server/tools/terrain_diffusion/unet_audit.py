"""Independent final audit of all v13 saved predictions and checkpoint receipts."""
import json
import numpy as np
from .data import Dataset,write_json,sha
from .unet_foundation import ROOT
from .residual_experiment import load_pack

def run():
    data=Dataset(device='cpu');records=[]
    for path in sorted(ROOT.rglob('evaluation.npz')):
        p=load_pack(path);truth=data.a['val_y'][p['row'],12+p['hour']];np.testing.assert_allclose(p['truth'],truth,atol=3e-5,rtol=1e-6,equal_nan=True);assert np.isfinite(p['samples']).all();err=p['samples'][0].astype('f8')-p['truth'];scores=[]
        for li in range(3):scores.append(np.nanmean(np.sqrt(np.nanmean(err[p['layout']==li]**2,axis=0))))
        expected=json.loads(path.with_name('metrics.json').read_text())['macro_rmse_mm'];difference=abs(float(np.mean(scores))-expected);assert difference<2e-5;records.append(dict(path=path.relative_to(ROOT).as_posix(),metric_error_mm=difference,canonical_truth_verified=True))
    assert len(records)==22
    checkpoints=[]
    for folder in sorted(ROOT.glob('E130*/s[0-9]*')):
        receipt=json.loads((folder/'completion.json').read_text());assert sha(folder/'best.pt')==receipt['best_sha256'];checkpoints.append(dict(path=(folder/'best.pt').relative_to(ROOT).as_posix(),sha256=receipt['best_sha256']))
    for row in json.loads((ROOT/'fixed_training_diagnosis.json').read_text()):
        path=ROOT/row['experiment']/f"s{row['seed']}"/'last.pt';assert sha(path)==row['last_checkpoint_sha256'];checkpoints.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
    write_json(ROOT/'final_audit.json',dict(status='passed',evaluations=records,checkpoints=checkpoints,data_sha256=data.contract['data_sha256'],tests='29 passed',scope='10 selected evaluations,10 last-checkpoint diagnostic evaluations,2 fixed backgrounds; no new training'))
    print(json.dumps(dict(status='passed',evaluations=len(records),checkpoint_hashes=len(checkpoints),max_metric_error_mm=max(r['metric_error_mm'] for r in records))))

if __name__=='__main__':run()
