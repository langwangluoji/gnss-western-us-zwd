"""Training-moment spatial/temporal Gaussian copula, matching Gaussian marginals."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import sha,write_json
from .native_weather_experiment import CachedData
from .native_weather_models import NativeGaussian


def spatial_correlation(static,rho):
    xy=static[:,:2]*1000
    distance=torch.cdist(xy,xy,compute_mode='donot_use_mm_for_euclid_dist')/300
    dh=(static[:,None,2]-static[None,:,2])*2
    kernel=torch.exp(-.5*(distance.square()+dh.square()))
    return (1-rho)*torch.eye(len(static),device=static.device)+rho*kernel


def correlate_noise(noise,static,rho,time_correlation):
    space=torch.linalg.cholesky(spatial_correlation(static,rho))
    time=torch.linalg.cholesky(time_correlation)
    return torch.einsum('qr,brh->bqh',space,noise)@time.T


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--embeddings',type=Path,required=True);p.add_argument('--gaussian',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.25)
    data=CachedData(args.cache)
    protocol=json.loads((args.gaussian/'protocol.json').read_text(encoding='utf-8'))
    if data.development or protocol['config']['domain']!='gaussian':raise ValueError('requires training Gaussian source')
    if protocol['embedding_manifest_sha256']!=sha(args.embeddings/'manifest.json'):raise ValueError('Gaussian embedding source differs')
    destination=args.gaussian/'copula.json'
    if destination.exists():raise FileExistsError('copula already frozen')
    stats=protocol['normalization'];bias=data.tensor(stats['bias_mm']);scale=data.tensor(stats['scale_mm'])
    model=NativeGaussian().cuda().eval();model.load_state_dict(torch.load(args.gaussian/'last.pt',weights_only=True))
    mean=np.load(args.embeddings/'mean.npy',mmap_mode='r');features=np.load(args.embeddings/'embedding.npy',mmap_mode='r')
    covariance=torch.zeros(7,7,device='cuda',dtype=torch.float64);count=0;spatial_num=0.;spatial_den=0.
    for i in range(len(data)):
        mu,logstd=model(data.tensor(features[i]));target=(data.tensor(data.a['truth'][i])-data.tensor(mean[i])-bias)/scale
        z=(target-mu)*(-logstd).exp()
        eligible=torch.isfinite(z).all(-1);flat=z[eligible].double()
        covariance+=flat.T@flat;count+=len(flat)
        query=data.tensor(data.a['query_ids'][i],torch.long);st=data.query_static[query]
        kernel=spatial_correlation(st,1);kernel.fill_diagonal_(0)
        clean=torch.where(eligible[:,:,None],z,0)
        products=torch.einsum('bqh,brh->qr',clean,clean)
        pair_count=torch.einsum('bq,br->qr',eligible.float(),eligible.float())*7
        spatial_num+=float((kernel*products).sum());spatial_den+=float((kernel.square()*pair_count).sum())
        if i==0 or (i+1)%500==0:print(json.dumps(dict(stage='copula_fit',completed=i+1,total=len(data))),flush=True)
    covariance/=count;diagonal=covariance.diag().clamp_min(1e-8).sqrt()
    correlation=covariance/diagonal[:,None]/diagonal[None]
    correlation=.95*correlation+.05*torch.eye(7,device='cuda')
    rho=float(np.clip(spatial_num/max(spatial_den,1e-8),0,.95))
    write_json(destination,dict(spatial_rho=rho,time_correlation=correlation.cpu().tolist(),
        spatial_kernel='Gaussian300km, height1500m; separable Gaussian correlation, unit marginals',
        method='training standardized-residual second moments;5%time shrinkage',
        training_blocks=len(data),training_records=count,gaussian_weights_sha256=sha(args.gaussian/'last.pt'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),code_sha256=sha(Path(__file__)),
        strict_oof=False,no_development_labels_used=True))


if __name__=='__main__':main()
