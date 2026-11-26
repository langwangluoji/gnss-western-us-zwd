"""Explicit evaluation amendment for label-empty chronological blocks; no training changes."""
from pathlib import Path
import numpy as np


AMENDMENT=Path(__file__).with_name('inducing_flow_evaluation_amendment.json')


def label_support(cache,seed):
    from campaign import read
    from tools.terrain_diffusion.data import sha
    manifest=read(cache/'manifest.json')
    for n in ('truth.npy','origins.npy'):
        if sha(cache/n)!=manifest['files'][n]:raise ValueError('training label/clock array changed')
    truth=np.load(cache/'truth.npy',mmap_mode='r');counts=[]
    for i in range(len(truth)):
        rng=np.random.default_rng(seed+261126+i);index=np.sort(rng.choice(truth.shape[2],32,replace=False))
        counts.append(int(np.isfinite(truth[i][:,index]).sum()))
    zero=np.flatnonzero(np.asarray(counts)==0)
    return dict(blocks=len(counts),supervised_blocks=int(np.count_nonzero(counts)),
        zero_label_block_indices_0_based=zero.tolist(),zero_label_block_numbers_1_based=(zero+1).tolist(),
        total_finite_labels=sum(counts),finite_labels_by_block=counts,truth_sha256=manifest['files']['truth.npy'],
        origins_sha256=manifest['files']['origins.npy'],cache_manifest_sha256=sha(cache/'manifest.json'))


def require_counts(status,final,curve,support):
    n=support['supervised_blocks']
    if support['blocks']!=3000 or status['steps']!=3000 or status['supervised_updates']!=n or final['step']!=3000:
        raise ValueError('chronological or supervised completion count differs')
    if final['scheduler']['last_epoch']!=n or any(int(v['step'])!=n for v in final['optimizer']['state'].values()):
        raise ValueError('optimizer/scheduler clock differs from actual label support')
    if curve['step']!=3000 or curve['supervised_updates']!=n or curve['total_labels']!=support['total_finite_labels']:
        raise ValueError('curve label/clock accounting differs')


def require_actual_completion(model,support):
    import torch
    from campaign import read
    final=torch.load(model/'checkpoint_3000.pt',map_location='cpu',weights_only=True)
    require_counts(read(model/'status.json'),final,read(model/'curve.json')[-1],support)


def validate_amendment():
    from campaign import read
    from inducing_flow_series import LOCK,source_paths
    from tools.terrain_diffusion.data import sha
    amendment=read(AMENDMENT)
    if amendment['original_lock_sha256']!=sha(LOCK):raise ValueError('original lock changed')
    for n,v in amendment['new_code_sha256'].items():
        if sha(Path(__file__).with_name(n))!=v:raise ValueError('amended evaluation code changed')
    for seed in (42,17):
        if label_support(source_paths(seed)['cache'],seed)!=amendment['label_support'][str(seed)]:
            raise ValueError('amended support differs from actual training labels')
    return amendment
