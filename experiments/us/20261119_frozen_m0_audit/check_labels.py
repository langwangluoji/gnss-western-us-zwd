from reproduce import *
from gnss_field.evaluation.metrics import point_metrics
with np.load(next((m/'eval_005000').glob('forecast_*.npz'))) as z: t=z['hours'][:1]
before=field(t);support=b.support(t[0])[1]
old_targets=store.targets
store.targets=lambda role,hours: np.full((len(hours),len(store.ids(role))),-9876.,dtype='f4')
changed=field(t)
assert torch.equal(before,changed) and np.array_equal(support,b.support(t[0])[1])
store.targets=old_targets
try:point_metrics(np.array([[1.]]),np.array([[np.nan]]),np.array([[True]]))
except ValueError:pass
else:raise RuntimeError('Scored NaN accepted')
print('PASS C label perturbation and nonfinite forecast rejection')
