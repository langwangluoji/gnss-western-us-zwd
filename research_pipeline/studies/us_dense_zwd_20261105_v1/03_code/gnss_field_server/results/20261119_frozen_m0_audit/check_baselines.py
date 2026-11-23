from reproduce import *
from gnss_field.training.baselines import get_baseline_predictor
from gnss_field.data import InputScenario
from types import SimpleNamespace
r=dict(np.load(root/'monthly/2019-01.npz'));errs=[]
for i in [0,240,600]:
 t=r['hours'][i:i+1];ctx=b.build_context(t);f=field(t)
 cache=SimpleNamespace(scenario_hash=InputScenario().identity,get=lambda hours:f)
 for key,name in [('coarse','coarse_bilinear'),('idw','A_IDW'),('innovation','M0_innovation_IDW')]:
  pred=get_baseline_predictor(name,store,mean_cache=cache)(t,ctx,ids,1)['mean'][0]
  e=float(np.max(np.abs(pred-r[key][i])));assert e<=.001,(key,e)
  errs.append((int(t[0]),name,e))
(root/'baseline_equivalence.json').write_text(json.dumps(errs,indent=2));print(errs)
