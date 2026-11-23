"""Post-run positive-grid output check; original predictions remain unchanged."""
import numpy as np
from operator_spike import ROOT
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from gnss_field.data import PackedStore,ContextBuilder
from gnss_field.evaluation import point_metrics
from local_context_features import VertexQuery
from grid_pair_data import local_runtime
from gnss_field.training.prediction import safe_query_geometry


def main():
    immutable_json(ROOT/'physical_output_plan.lock.json',dict(code=sha256(__file__),
        rule='Z0 = temperature * logaddexp(0, raw_Z0 / temperature), before Q',temperature_mm=.1,
        scope='Post-run engineering parameterization, all methods equally; no retuning, no new independent confirmation',
        masks='original F/S0 unchanged; all land retained with support flag',server_jobs=False))
    # The temperature is fixed in physical units, not fitted from scores.
    grid=np.array([-1000.,-1.,0.,1.,1000.]);positive=.1*np.logaddexp(0,grid/.1)
    assert np.isfinite(positive).all() and (positive>=0).all() and abs(positive[-1]-1000)<1e-10
    store=PackedStore(local_runtime()['data_root'],purpose='context');builder=ContextBuilder(store)
    land=store.grid['region'].ravel().astype(bool);output={}
    for directory in ('C2019_confirmation','C2019_replication'):
        dest=ROOT/directory;raw=dict(np.load(dest/'published_fields.npz'));inputs=np.load(dest/'inputs.npz')
        fields={n:(.1*np.logaddexp(0,v.astype('f8')/.1)).astype('f4') for n,v in raw.items() if n!='hours'}
        support=[]
        for valid in inputs['valid']:
            _,geo=builder._geometry(valid.tobytes(),'grid');support.append(geo['support']&builder.label_near_grid&land)
        np.savez_compressed(dest/'positive_fields.npz',hours=raw['hours'],land=land,support=np.array(support),**fields)
        z=dict(np.load(dest/'station_predictions.npz'));xy,height=safe_query_geometry(store,z['station_ids'])
        xy=np.minimum(xy,np.array([95-1e-5,143-1e-5]));q=VertexQuery(store.shape,xy,height,np.arange(np.prod(store.shape)))
        rows={}
        for name,v in fields.items():
            assert np.isfinite(v).all() and (v>=0).all()
            p=(v[:,q.indices.numpy()]*q.weights.numpy()[None]).sum(2)
            rows[name]=dict(score=point_metrics(z['truth'],p,z['population']),
                scored_prediction_change_max_mm=float(abs(p-z[name])[z['population']].max()),negative_land_grid_hours=0)
        output[directory]=rows
    write_json(ROOT/'physical_output_checks.json',dict(passed=True,rows=output,
        interpretation='Physical nonnegativity checked; unsupported-field accuracy remains unvalidated; original raw metrics retained'))
    print('POSITIVE_OUTPUT_VERIFIED',{d:{n:round(v['scored_prediction_change_max_mm'],7) for n,v in rows.items()} for d,rows in output.items()},flush=True)


if __name__=='__main__':main()
