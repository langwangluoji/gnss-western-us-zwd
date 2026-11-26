import unittest
import torch
import numpy as np
from field_generation_v1.training import save_checkpoint,load_checkpoint,update_ema,next_steps
from tempfile import TemporaryDirectory
from pathlib import Path

class RestoreTests(unittest.TestCase):
    def test_pending_evaluation_replayed_before_next_update(self):
        meta=dict(pending_evaluation=True,training_finished=False,budget=30000)
        self.assertEqual(next(iter(next_steps(1000,meta))),1000)
        meta['pending_evaluation']=False
        self.assertEqual(next(iter(next_steps(1000,meta))),1001)
    def test_reranking_resume_never_restarts_optimizer(self):
        self.assertEqual(list(next_steps(7000,dict(pending_evaluation=False,training_finished=True,budget=30000))),[])
    def test_100_resume_100_equals_200(self):
        def create():
            torch.manual_seed(42);m=torch.nn.Linear(3,1);e=torch.nn.Linear(3,1);e.load_state_dict(m.state_dict())
            return m,e,[torch.optim.AdamW(m.parameters(),lr=.001)],np.random.default_rng(42)
        def steps(m,e,opts,rng,n):
            for _ in range(n):
                x=torch.randn(8,3)+float(rng.normal());loss=(m(x)-x.sum(1,keepdim=True)).square().mean()
                opts[0].zero_grad();loss.backward();opts[0].step();update_ema(e,m)
        a,ae,ao,ar=create();steps(a,ae,ao,ar,200)
        b,be,bo,br=create();steps(b,be,bo,br,100)
        with TemporaryDirectory() as d:
            p=Path(d)/'last.pt';save_checkpoint(p,b,be,bo,br,100,{'cursor':100})
            c,ce,co,cr=create();state=load_checkpoint(p,c,ce,co,cr);steps(c,ce,co,cr,100)
        self.assertEqual(state['step'],100)
        for key,value in a.state_dict().items():torch.testing.assert_close(value,c.state_dict()[key],rtol=0,atol=0)
        for key,value in ae.state_dict().items():torch.testing.assert_close(value,ce.state_dict()[key],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
