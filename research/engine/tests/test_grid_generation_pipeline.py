import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_outer_pilot_hours_do_not_depend_on_labels():
 from grid_generation_data import pilot_outer_hours
 hours=np.concatenate([np.arange(1000+200*i,1168+200*i) for i in range(12)])
 chosen=pilot_outer_hours(hours)
 assert len(chosen)==96 and np.isin(chosen,hours).all()
 assert np.array_equal(chosen[:8],1000+np.arange(10,168,21))
 with pytest.raises(ValueError): pilot_outer_hours(hours[:-1])


def test_probabilistic_metrics_fail_on_nonfinite_and_report_widths():
 from grid_generation_metrics import score_ensemble
 rng=np.random.default_rng(42); truth=rng.normal(size=(5,8)); samples=rng.normal(size=(5,16,8))
 mask=np.ones((5,8),bool); xy=np.c_[np.arange(8)*1000,np.zeros(8)]
 m=score_ensemble(truth,samples,truth,mask,xy)
 assert m['station_macro_rmse_mm']==0
 assert 0<m['coverage90']<=1 and m['width90_mm']>0
 assert np.isfinite(m['group_ES_mm']) and np.isfinite(m['group_VS'])
 samples[0,0,7]=np.nan; mask[0,7]=False
 with pytest.raises(ValueError): score_ensemble(truth,samples,truth,mask,xy)

@pytest.mark.parametrize('kind',['stochastic','diffusion','gan'])
def test_actual_engine_recovers_generator_critic_optimizer_and_rng(tmp_path,kind):
 import torch
 from types import SimpleNamespace
 from grid_generation_data import GenerationData
 from grid_generation_train import make_engine
 from gnss_field.artifacts import write_json
 from gnss_field.training.observable import _same
 torch.set_num_threads(1); torch.manual_seed(11)
 class Toy(GenerationData):
  def __init__(self):
   self.root=tmp_path; self.device='cpu'; self.deadline=None; self.shape=(8,8)
   self.train_hours=np.arange(16); self.folds=np.array([0]*4+[1]*4); self.ids=np.arange(8)
   self.coords=torch.randn(8,3); self.xy=self.coords[:,:2].numpy()*400000
   self.phi=torch.randn(64,7)*.3; self.v=torch.linalg.qr(torch.randn(7,3)).Q.T; self.rank=3; self.rest=7; self.tau=.1
   self.x=torch.randn(16,69,8,8); self.y=torch.randn(16,8)
   indices=torch.tensor([[i,i+1,i+8,i+9] for i in range(8)])
   self.source=SimpleNamespace(qindices=indices,qweights=torch.full((8,4),.25),indices=lambda h:np.asarray(h),t={'truth':self.y,'valid':torch.ones(16,8,dtype=torch.bool)})
   self.geometry_weights=self.source.qweights
   self.a={'valid':np.ones((16,8),bool)}
  def batch(self,hours,fold):
   ix=np.asarray(hours); return ix,self.x[ix],torch.zeros(len(ix),8),torch.ones(len(ix)),torch.ones(len(ix),dtype=torch.bool)
 data=Toy(); write_json(tmp_path/'data.lock.json',{'toy':42})
 def engine(folder): return make_engine(data,kind,0,tmp_path/folder,engineering_hours=data.train_hours[:8],widths=(8,16,16,16))
 a,obj,before,_=engine('continuous'); a.run(obj,before_update=before,stop_after=12); expected=a.state()
 b,obj,before,_=engine('split'); b.run(obj,before_update=before,stop_after=6)
 b,obj,before,_=engine('split'); b.run(obj,before_update=before,stop_after=12); actual=b.state()
 for key in ('model','ema','optimizers','auxiliary_optimizers','cursor','rng','step'):
  _same(expected[key],actual[key])
 assert not torch.equal(actual['model']['generator.net.output.weight'],torch.zeros_like(actual['model']['generator.net.output.weight']))


def test_resume_budget_accounts_completed_training_and_engineering(tmp_path):
 from types import SimpleNamespace
 from grid_generation_train import remaining_estimate,ARMS
 from gnss_field.artifacts import write_json
 data=SimpleNamespace(root=tmp_path)
 b={'timings':{k:{'update_seconds':1.,'evaluation_seconds_per_hour':{'16':0.,'32':0.}} for k in ARMS}}
 assert remaining_estimate(data,b)==1.5*3*(2*684+400)
 write_json(tmp_path/'engineering/verification.json',{'passed':True})
 for fold in (0,1):
  for kind in ARMS: write_json(tmp_path/f'fold_{fold}/{kind}/results.json',{'complete':True})
 assert remaining_estimate(data,b)==0


def test_pause_is_only_checked_inside_evaluation_boundary(tmp_path):
 from grid_generation_data import GenerationData
 from gnss_field.training.engine import PauseRequested
 d=object.__new__(GenerationData); d.root=tmp_path; d.deadline=None
 (tmp_path/'PAUSE').touch()
 d.check()
 with pytest.raises(PauseRequested): d.check(force=True)


def test_release_refuses_missing_real_mean_artifacts(tmp_path,monkeypatch):
 import grid_generation_release as release
 monkeypatch.setattr(release,'PROJECT',tmp_path)
 with pytest.raises(ValueError,match='verified downloaded old results'):
  release.package()


def test_partial_benchmark_resume_measures_new_updates():
 from grid_generation_train import measure_updates
 class Engine:
  step=5
  def run(self,objective,before_update=None,stop_after=None):
   self.step=stop_after
   return {'status':'PAUSED','step':self.step}
 e=Engine()
 ticks=iter([10.,13.])
 row=measure_updates(e,None,None,synchronize=lambda:None,clock=lambda:next(ticks))
 assert e.step==10
 assert row['updates_measured']==3 and row['start_step']==7 and row['end_step']==10
 assert row['update_seconds']>0


def test_pending_evaluation_and_product_selection_restore(tmp_path):
 import torch
 from gnss_field.training.engine import TrainEngine,StageSpec,PauseRequested
 from gnss_field.contracts import ScienceConfig
 from grid_generation_train import initialize
 torch.set_num_threads(1)
 science=ScienceConfig(ema=.99,warmup=1,startup_evaluation=2,evaluate_every=3,log_every=1,minimum_refinement_steps=6)
 def create(name):
  torch.manual_seed(42)
  return TrainEngine(torch.nn.Linear(1,1),tmp_path/name,np.arange(8),StageSpec('generation_refine',6,6,2,2,42,{'toy':True}),science)
 def objective(hours,model,step):
  x=torch.tensor(hours,dtype=torch.float32)[:,None]/8
  return {'loss':(model(x)-torch.randn_like(x)).square().mean()}
 def evaluator(model,step):
  v=float(next(model.parameters()).square().sum())
  return {'point':{'value':v},'joint':{'value':v},'marginal':{'value':v}}
 a=create('continuous'); initialize(a,evaluator); a.run(objective,evaluator=evaluator)
 b=create('split'); initialize(b,evaluator)
 def fail_once(model,step):
  if step==3: raise PauseRequested('Simulated interrupted evaluation')
  return evaluator(model,step)
 with pytest.raises(PauseRequested): b.run(objective,evaluator=fail_once)
 b=create('split'); assert b.pending_evaluation and b.step==3
 initialize(b,evaluator); b.run(objective,evaluator=evaluator)
 assert not b.pending_evaluation and b.step==6
 from gnss_field.training.observable import _same
 _same(a.model.state_dict(),b.model.state_dict()); _same(a.ema.state_dict(),b.ema.state_dict())
 for product in ('point','joint','marginal'):
  assert [(x['step'],x['value']) for x in a.products[product]]==[(x['step'],x['value']) for x in b.products[product]]
