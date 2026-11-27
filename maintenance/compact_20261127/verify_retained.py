"""Verify protected scientific bytes and core weight deserialization; no training."""
import collections
import json
from pathlib import PurePosixPath

import torch
import numpy as np

from compact import HERE, ROOTS, extended, read_rows, safe_path, sha


def load_weights(path):
    # Original fixed CNN checkpoints store ndarray normalization beside tensors.
    allowed=[np.ndarray,np.dtype,np._core.multiarray._reconstruct,np._core.multiarray.scalar]
    allowed += [type(np.dtype(t)) for t in (np.float64,np.float32,np.int64,np.int32,np.uint32,np.bool_)]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location='cpu',weights_only=True)


def main():
    torch.set_num_threads(2)
    rows=read_rows(); failures=[]; protected=[]; removed=[]; loaded=[]
    for row in rows:
        path=safe_path(ROOTS[row['scope']],row['path'])
        if row['action'] in {'archive_model','retire_derived'}:
            if path.exists():failures.append({'path':row['path'],'problem':'retired file still exists'})
            removed.append(row)
            continue
        # Documentation/implementation edits are intentional; scientific data aren't.
        if row['scope']=='project' and not row['path'].startswith('artifacts/'):
            continue
        if not path.is_file() or path.stat().st_size!=row['size'] or sha(path)!=row['sha256']:
            failures.append({'scope':row['scope'],'path':row['path'],'problem':'protected bytes changed'})
        else:
            protected.append(row)
        if row['action']=='keep_core' and PurePosixPath(row['path']).suffix in {'.pt','.pth'}:
            state=load_weights(path)
            loaded.append({'path':row['path'],'sha256':row['sha256'],'type':type(state).__name__,
                           'keys':sorted(map(str,state.keys())) if isinstance(state,dict) else []})
            del state
        if len(protected)%500==0 and protected:
            print(json.dumps({'protected_hashed':len(protected),'core_weights_loaded':len(loaded)}),flush=True)
    log=[json.loads(line) for line in (HERE/'deletion_log.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    actual={(r['scope'],r['path'],r['sha256']) for r in log}
    expected={(r['scope'],r['path'],r['sha256']) for r in removed}
    if actual!=expected or len(log)!=len(removed):failures.append({'problem':'deletion log differs from inventory'})
    result={'status':'passed' if not failures else 'failed','protected_files':len(protected),
            'protected_bytes':sum(r['size'] for r in protected),'core_weight_files_loaded':len(loaded),
            'deleted_files':len(removed),'deleted_bytes':sum(r['size'] for r in removed),
            'weights':loaded,'failures':failures,'training_started':False,
            'target_values_parsed':False,'confirmation_labels_parsed':False,
            'inventory_sha256':sha(HERE/'inventory.csv')}
    (HERE/'retention_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'weights'}}),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
