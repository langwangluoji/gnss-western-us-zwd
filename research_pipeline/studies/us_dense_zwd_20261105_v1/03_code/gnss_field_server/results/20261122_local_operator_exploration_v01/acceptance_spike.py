"""Independent numerical checks and paired summaries for this local spike."""
from pathlib import Path
import numpy as np
import torch
from operator_spike import ROOT,SOURCE,query,lattice_kernel
from gnss_field.artifacts import read_json,write_json,sha256
from gnss_field.data import PackedStore
from gnss_field.evaluation import point_metrics
from grid_pair_data import local_runtime
from local_context_features import VertexQuery
from gnss_field.training.prediction import safe_query_geometry,safe_fine_sites
from gnss_field.geometry import coarsen_100km
from generation_postrun_diagnostics import paired_block_rmse
from local_innovation_report import paired_bootstrap


def main():
    torch.set_num_threads(2);store=PackedStore(local_runtime()['data_root'],purpose='context')
    a=dict(np.load(SOURCE/'data.npz'));old=read_json(SOURCE/'results.json');new=read_json(ROOT/'results.json')
    reproducibility=max(abs(new[f]['rows']['linear'][role]['station_macro_rmse_mm']-old['folds'][f]['rows']['linear'][role]['station_macro_rmse_mm']) for f in ('0','1') for role in ('inner','outer'))
    assert reproducibility<.001
    inputs=dict(np.load(ROOT/'A_inputs.npz'))
    for name in ('raw','valid','residual','base_A'):inputs[name]=inputs[name][:4]
    order=np.random.default_rng(42).permutation(len(inputs['A_xy']));permuted={}
    for k,v in inputs.items():permuted[k]=v[:,order] if k in ('raw','valid','residual','base_A') else v[order]
    nodes=a['global_nodes'];one=lattice_kernel(inputs,store,nodes,.1,750.)
    two=lattice_kernel(permuted,store,nodes,.1,750.)
    permutation_error=float(abs(one-two).max());assert permutation_error<1e-5
    missing=dict(inputs);missing['valid']=np.zeros_like(inputs['valid'])
    assert np.array_equal(lattice_kernel(missing,store,nodes,.1,750.),np.zeros_like(one))
    missing['valid'][:,0]=True
    assert np.isfinite(lattice_kernel(missing,store,nodes,.1,750.)).all()
    lock=read_json(SOURCE/'experiment.lock.json');oi=np.searchsorted(a['hours'],lock['outer_hours'])
    audit=read_json(Path(local_runtime()['outer']));weeks=np.array([np.datetime64(s,'h').astype('i8') for s in audit['week_starts']])
    intervals={}
    for f,split in enumerate(lock['splits']):
        p=dict(np.load(ROOT/f'fold_{f}_predictions.npz'));p['gate']=np.load(ROOT/f'gate_fold_{f}/prediction.npz')['prediction']
        mask=a['valid'][oi]&np.asarray(split['outer'],bool)[None]
        intervals[str(f)]={name:paired_bootstrap(a['truth'][oi],p[name][oi],a['mean'][oi],mask,a['hours'][oi],weeks,a['xy']) for name in ('directional','kernel','gate')}
    write_json(ROOT/'B_paired_intervals.json',dict(comparisons=intervals,scope='12 sampled weather clusters; exploratory, no multiplicity adjustment'))
    reload_checks={}
    for directory in ('C2019_confirmation','C2019_replication'):
        dest=ROOT/directory
        z=dict(np.load(dest/'station_predictions.npz'));fields=dict(np.load(dest/'published_fields.npz'))
        assert sha256(dest/'published_fields.npz')==read_json(dest/'prediction_commit.json')['fields_sha256']
        xy,height=safe_query_geometry(store,z['station_ids']);xy=np.minimum(xy,np.array([95-1e-5,143-1e-5]))
        q=VertexQuery(store.shape,xy,height,np.arange(np.prod(store.shape)))
        reload_error=max(float(abs(query(field,q.indices.numpy(),q.weights.numpy())-z[name]).max()) for name,field in fields.items() if name!='hours')
        assert reload_error<.001;reload_checks[directory]=reload_error
        ci={}
        for name in ('linear0','linear1','directional0','directional1','kernel0','kernel1','gate0','gate1'):
            ci[name]=paired_block_rmse(z['truth'],z['base'],z[name],z['population'],np.arange(48)//4)
            ci[name]['interpretation']='candidate minus base; 12 sampled monthly days, 4 hours each; unadjusted exploratory interval'
        write_json(dest/'paired_intervals.json',dict(comparisons=ci,scope='not a complete-week or annual bootstrap'))
        mask=z['population']&safe_fine_sites(store,z['station_ids'])[None]
        fine={}
        for name,field in fields.items():
            if name=='hours':continue
            coarse=coarsen_100km(torch.tensor(field.reshape(48,*store.shape))).reshape(48,-1)
            fine[name]=dict(original=point_metrics(z['truth'],z[name],mask),coarsened=point_metrics(z['truth'],q(coarse).numpy(),mask))
        write_json(dest/'fine_comparison.json',dict(rows=fine,population='F & S0 & fixed T complete land-safe coarse templates'))
    write_json(ROOT/'numerical_acceptance.json',dict(passed=True,linear_reproduction_max_mm=reproducibility,
        input_station_permutation_max_mm=permutation_error,zero_and_single_A=True,published_reload_max_mm=reload_checks,
        code_scope='Throwaway local spike; production code unchanged; no new full-suite claim',server_jobs=False))
    print('NUMERICAL_ACCEPTANCE_PASSED',reproducibility,permutation_error,reload_checks,flush=True)


if __name__=='__main__':main()
