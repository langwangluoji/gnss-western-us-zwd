"""Exploratory spatial-member shuffle; every station marginal stays identical."""
import argparse
import json
import numpy as np
from scipy.spatial.distance import cdist
from .data import Dataset,write_json
from .transport_train import ROOT
from .metrics import evaluate


def run(a):
    data=Dataset(device='cpu');dist=cdist(data.a['val_geo'][:,:2],data.a['val_geo'][:,:2]);i,j=np.triu_indices(137,1)
    groups={'0to50km':(dist[i,j]>0)&(dist[i,j]<=50),'50to150km':(dist[i,j]>50)&(dist[i,j]<=150)}
    results=[]
    for seed in (42,17):
        for arm in ('gaussian','diffusion'):
            name=f'B_{arm}_s{seed}';folder=ROOT/name if a.scope=='selection' else ROOT/name/a.scope
            pack=np.load(folder/'evaluation.npz');values={label:dict(original=[],shuffled=[[] for _ in range(a.repeats)]) for label in groups}
            for layout in range(3):
                s=pack['samples'][layout];y=pack['truth'][layout]
                for label,keep in groups.items():
                    left,right=i[keep],j[keep]
                    values[label]['original'].append(evaluate(s[:,:,left]-s[:,:,right],y[:,left]-y[:,right])['fair_crps_mm'])
                for repeat in range(a.repeats):
                    rng=np.random.default_rng(88000+repeat*100+layout);shuffled=s.copy()
                    for t in range(s.shape[1]):
                        for station in range(s.shape[2]):shuffled[:,t,station]=s[rng.permutation(s.shape[0]),t,station]
                    np.testing.assert_array_equal(np.sort(shuffled,axis=0),np.sort(s,axis=0))
                    for label,keep in groups.items():
                        left,right=i[keep],j[keep]
                        values[label]['shuffled'][repeat].append(evaluate(shuffled[:,:,left]-shuffled[:,:,right],y[:,left]-y[:,right])['fair_crps_mm'])
            for label,v in values.items():
                original=float(np.mean(v['original']));shuffled=[float(np.mean(x)) for x in v['shuffled']]
                results.append(dict(run=name,distance_group=label,pairs=int(groups[label].sum()),original_crps_mm=original,
                                    shuffle_crps_mm=shuffled,shuffle_mean_minus_original_mm=float(np.mean(shuffled)-original)))
            print(json.dumps(dict(spatial_done=name,scope=a.scope)),flush=True)
    write_json(ROOT/f'spatial_shuffle_{a.scope}.json',dict(scope=a.scope,repeats=a.repeats,records=results,
        status='exploratory_posthoc_no_checkpoint_selection',station_marginals_exactly_preserved=True,
        meaning='CRPS of observed station-to-station ZWD differences, not point ZWD RMSE or unobserved grid accuracy'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',default='selection');p.add_argument('--repeats',type=int,default=5);run(p.parse_args())
