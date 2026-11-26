"""Input-isolation checks before training; trained grid/sequence checks after."""
import json
import sys
import numpy as np
import torch
from scipy.spatial.distance import cdist
import masked_graph_spike as m
from gnss_field.artifacts import write_json,sha256


@torch.no_grad()
def main():
    torch.set_num_threads(2);a=dict(np.load(m.old.exp.ROOT/'data.npz'));g=m.geometry(a);d=m.dataset(a,'inner')
    torch.manual_seed(42);model=m.GraphField().cuda()
    post='--post' in sys.argv
    if post:model.load_state_dict(torch.load(m.ROOT/'masked_42.pt',weights_only=False)['model'])
    # Nonzero outputs in the preflight make isolation checks substantive.
    else:
        torch.nn.init.normal_(model.prior[-1].weight,std=.1);torch.nn.init.normal_(model.obs[-1].weight,std=.1)
    hidden=torch.zeros(174,dtype=torch.bool,device='cuda');hidden[:44]=True;ids=a['aids'][:44]
    p=m.predict(model,d,g,[0],ids,hidden);dp=dict(d);dp['raw']=d['raw'].clone();dp['raw'][:,:,:44]=99999
    hidden_error=float((p-m.predict(model,dp,g,[0],ids,hidden)).abs().max());assert hidden_error==0
    dp=dict(d);dp['raw']=d['raw'].clone();dp['raw'][:,18:]=99999
    causal_error=float((p[:,:12]-m.predict(model,dp,g,[0],ids,hidden)[:,:12]).abs().max());assert causal_error==0
    chunks=torch.cat([m.predict(model,d,g,[0],part,hidden) for part in np.array_split(ids,4)],-1)
    chunk_error=float((p-chunks).abs().max());assert chunk_error<1e-4
    perm=torch.randperm(174,device='cuda');reverse=torch.argsort(perm);gp=dict(g)
    gp['neighbors']=reverse[g['neighbors'][perm]];gp['edges']=g['edges'][perm];gp['qn']=reverse[g['qn']]
    dp=dict(d);dp['raw']=d['raw'][:,:,perm]
    perm_error=float((p-m.predict(model,dp,gp,[0],ids,hidden[perm])).abs().max());assert perm_error<1e-4
    checks=dict(hidden_value_perturbation_mm=hidden_error,future_value_perturbation_mm=causal_error,query_chunk_mm=chunk_error,node_permutation_mm=perm_error)
    if post:
        lock=json.loads((m.ROOT/'experiment.lock.json').read_text());result=json.loads((m.ROOT/'results.json').read_text())
        assert sha256(m.__file__)==lock['code_sha256']==result['runtime']['code_sha256']
        assert sha256(m.ROOT/'PROTOCOL.md')==lock['protocol_sha256']
        assert sha256(m.old.exp.ROOT/'data.npz')==lock['data_sha256']
        assert sha256(m.ROOT/'selection.lock.json')==result['runtime']['selection_sha256']
        for name,h in lock['dependencies'].items():assert sha256(m.core.PROJECT/'tools'/name)==h
        for name,h in result['runtime']['checkpoints'].items():assert sha256(m.ROOT/name)==h
        # Build all published grid nodes; retain one spatially shared noise draw.
        ax=a['xy'][a['aids']];ah=a['heights'][a['aids']];xy=m.old.XY;terrain=m.old.TERRAIN
        dist=cdist(xy,ax)/1000;qn=np.argsort(dist,axis=1)[:,:12]
        gf=dict(g);gf['qn']=torch.tensor(qn,device='cuda');gf['qweight']=torch.tensor(np.exp(-.5*(np.take_along_axis(dist,qn,axis=1)/150)**2),dtype=torch.float32,device='cuda')
        qe=np.concatenate([(ax[qn]-xy[:,None])/150000,(ah[qn]-terrain[:,0,None]*3000)[...,None]/750],-1)
        gf['qedge']=torch.tensor(qe,dtype=torch.float32,device='cuda');gf['qstatic']=torch.tensor(terrain,dtype=torch.float32,device='cuda')
        gen=m.LatentField(True).cuda();gen.load_state_dict(torch.load(m.ROOT/'masked_42_ar.pt',weights_only=False)['model'])
        noise=torch.randn(1,8,24,8,device='cuda');means=[];fields=[]
        for start in range(0,len(xy),256):
            nodes=torch.arange(start,min(start+256,len(xy)),device='cuda');z,h=model(d['raw'][:1],d['cycles'][:1],gf,nodes)
            means.append(z.cpu());fields.append((z[:,None]+gen(h,noise)).cpu())
        mean=torch.cat(means,-1).numpy();field=torch.cat(fields,-1).numpy()
        ids=a['split_inner'];direct,h,w=m.predict(model,d,g,[0],ids,features=True)
        back=(mean[:,:,m.old.QI[ids]]*m.old.QW[ids]).sum(-1)*np.exp(-a['heights'][ids]/3000)
        error=float(np.max(np.abs(back-direct.cpu().numpy())));assert error<1e-4
        cached=dict(mu=direct,h=h,w=w,height=d['height'][ids]);query=m.samples(gen,cached,noise).cpu().numpy()
        back=(field[:,:,:,m.old.QI[ids]]*m.old.QW[ids]).sum(-1)*np.exp(-a['heights'][ids]/3000)
        sample_error=float(np.max(np.abs(back-query)));assert sample_error<1e-4
        assert np.isfinite(field).all()
        np.savez_compressed(m.ROOT/'example_grid_sequence.npz',z0=field.reshape(8,24,144,96),mean_z0=mean.reshape(24,144,96),hours=a['inner_hours'][0,6:])
        checks.update(source_hashes=True,grid_mean_roundtrip_mm=error,grid_sample_roundtrip_mm=sample_error,sequence_shape=[8,24,144,96])
    write_json(m.ROOT/('verification.json' if post else 'preflight.json'),dict(checks=checks,verifier_sha256=sha256(__file__)))
    print(json.dumps(checks))


if __name__=='__main__':main()
