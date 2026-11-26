"""Real-data unit/field/gradient gates before any registered training."""
import copy
import json
import time
import os
import numpy as np
import torch
from field_generation_v1.data import FieldData
from field_generation_v1.model import MeanField,ResidualField,gaussian_losses
from field_generation_v1.training import Predictor
from field_generation_v1.geometry import query_grid,curvature
from field_generation_v1.prepare import ROOT
from residual_edm_v1.data import write_json

def main():
    from field_generation_v1.test_data_protocol import DataProtocolTests
    protocol=DataProtocolTests()
    protocol.test_dense_groups_are_indivisible()
    protocol.test_identity_coverage_and_input_lattice()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2);torch.manual_seed(42)
    data=FieldData('cuda');data.prepare_qualification()
    s=next(int(s) for s in data.train if len(data.qualified_hours(s,2016)))
    t=int(data.qualified_hours(s,2016)[0]);ids=data.patch_ids(s);b=data.batch(t,ids)
    base=MeanField().cuda().eval();p=Predictor(data,base)
    with torch.no_grad():
        full,_,off=p.predict_field(t);local,_,offset=p.predict_field(t,s)
        a,_=p.query(t,full,None,off,[s]);z,_=p.query(t,local,None,offset,[s])
        torch.testing.assert_close(a,z,rtol=0,atol=.001)
        saved=full.cpu().numpy();np.save(ROOT/'smoke_field.npy',saved)
        reloaded=torch.as_tensor(np.load(ROOT/'smoke_field.npy'),device='cuda')
        zz,_=p.query(t,reloaded,None,off,[s]);torch.testing.assert_close(a,zz,rtol=0,atol=.001)
        xy=torch.as_tensor(data.qxy[[s]],device='cuda',dtype=torch.float32)
        z0=query_grid(full,xy)[:,0];torch.testing.assert_close(a,z0*float(np.exp(-data.height[s]/3000)))
    torch.manual_seed(42);model=ResidualField().cuda().eval()
    with torch.no_grad():
        r,f=model(b);perm=torch.randperm(b['stations'].shape[1],device='cuda');other=dict(b,stations=b['stations'][:,perm],padding=b['padding'][:,perm])
        rr,ff=model(other);torch.testing.assert_close(f,ff,rtol=1e-4,atol=1e-5)
        missing=dict(b,padding=torch.ones_like(b['padding']));_,fm=model(missing);assert torch.isfinite(fm).all()
    # Several real-data updates: identical MSE/faithful mean trajectories despite arbitrary variance learning.
    left=copy.deepcopy(model).train();right=copy.deepcopy(model).train()
    def groups(m):return [p for n,p in m.named_parameters() if not n.startswith('variance_head.')]
    lo=torch.optim.AdamW(groups(left),lr=1e-4,weight_decay=1e-4)
    ro=torch.optim.AdamW(groups(right),lr=1e-4,weight_decay=1e-4)
    vo=torch.optim.AdamW(right.variance_head.parameters(),lr=1e-4,weight_decay=1e-4)
    target=torch.tensor([.3],device='cuda')
    for _ in range(3):
        for m,opt,faithful in [(left,lo,False),(right,ro,True)]:
            opt.zero_grad();vo.zero_grad();r,f=m(b);mean=r.mean().reshape(1)
            features=f.mean((0,2,3))[None];sigma=m.standard_deviation(features,features.new_zeros(1,3),1,True)
            mse,nll=gaussian_losses(mean,target,sigma.square(),1,True)
            loss=mse+nll if faithful else mse;loss.backward();torch.nn.utils.clip_grad_norm_(groups(m),1);opt.step()
            if faithful:torch.nn.utils.clip_grad_norm_(m.variance_head.parameters(),1);vo.step()
        for l,r in zip(groups(left),groups(right)):torch.testing.assert_close(l,r,atol=0,rtol=0)
    # Input construction reads only input values: changing held-out observations is structurally excluded.
    original=data.values
    class PerturbedLabels:
        def __getitem__(self,key):
            value=np.array(original[key],copy=True)
            # Exercise both training-time and inference-time calls: any non-input
            # read is intentionally poisoned, while input preparation must stay independent.
            return np.full_like(value,np.nan)
    before=data.batch(t,ids);old_stats=json.dumps(data.stats,sort_keys=True);before_support=data.support(t)[0].copy()
    data.values=PerturbedLabels();after=data.batch(t,ids)
    for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    np.testing.assert_array_equal(before_support,data.support(t)[0]);assert old_stats==json.dumps(data.stats,sort_keys=True)
    data.values=original
    assert not np.intersect1d(data.inputs,data.dev).size
    assert not np.intersect1d(data.train,data.dev).size
    write_json(ROOT/'smoke.json',dict(passed=True,time=time.time(),station=s,hour=t,field_shape=list(full.shape),
        contract=data.contract,real_local_global_mm=float((a-z).abs().max()),tests=['field disk reload','height restoration','station permutation','all missing','faithful mean trajectory','label poisoning input invariance']))
    print('REAL DATA SMOKE PASS',flush=True)

if __name__=='__main__':main()
