from pathlib import Path
import hashlib,json,sys, numpy as np,torch
from gnss_field.data import PackedStore, ContextBuilder
from gnss_field.models.mean import MeanField
from gnss_field.training.fields import predict_patches
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.geometry import surface_query
from gnss_field.evaluation.metrics import point_metrics
root=Path(__file__).resolve().parent
m=root/'release/seed_42/mean_base'
store=PackedStore('../../02_data/processed/gnss_field_server_v1',purpose='development')
b=ContextBuilder(store); ids=store.ids('C');xy,h=safe_query_geometry(store,ids)
ck=next((root.parent/'20261118_us_dense_zwd_a800_m0_5k_evalfix_v02').rglob('candidate_005000.pt'))
assert hashlib.sha256(ck.read_bytes()).hexdigest()==json.loads((m/'frozen.json').read_text())['checkpoint_sha256']
device='cuda' if torch.cuda.is_available() else 'cpu'
from gnss_field.runtime import configure
from types import SimpleNamespace
configure(SimpleNamespace(cpus=2))
model=MeanField().to(device).eval();model.load_state_dict(torch.load(ck,map_location=device,weights_only=False)['ema'])
def field(hours):
 with torch.inference_mode():return predict_patches(model,b.build_context(hours).to(device),store,chunk=8)[0].cpu()
errors=[]
for p in sorted((m/'eval_005000').glob('forecast_*.npz')):
 z=np.load(p);t=z['hours'][:1]; assert np.array_equal(ids,z['station_ids'])
 f=field(t);pred=surface_query(f,xy,h).numpy().reshape(-1);errors.append(float(np.max(abs(pred-z['prediction'][0]))))
print('Monthly errors',errors,flush=True)
assert len(errors)==12 and max(errors)<=.001
rows=[dict(np.load(p)) for p in sorted((m/'full2019/5000').glob('forecast_*.npz'))]
def cat(k):return np.concatenate([r[k] for r in rows])
truth=store.targets('C',cat('hours'));np.testing.assert_equal(truth,cat('truth'))
mask=cat('population_F')&cat('normal_support_S0');met=point_metrics(truth,cat('prediction'),mask)
assert met['n_stations']==76 and met['n_station_hours']==642258 and round(met['station_macro_rmse_mm'],3)==10.345
(root/'reproduction.json').write_text(json.dumps(dict(device=device,monthly_errors=errors,primary=met),indent=2))
print(met,flush=True)

