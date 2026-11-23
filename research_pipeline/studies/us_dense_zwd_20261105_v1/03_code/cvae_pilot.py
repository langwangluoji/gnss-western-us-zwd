"""Matched-fold conditional VAE pilot with strictly prior-only evaluation."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from us_gan import USResidualUNet, sample_points, valid_indices
from train_us_gan import load, batch, station_arrays, masked_mse
from feature_learning import development_truth, evaluate

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / '04_models/cvae_pilot_v1'

def gaussian_kl(mean_q, logvar_q, mean_p, logvar_p):
    return .5 * (logvar_p - logvar_q + (logvar_q.exp() + (mean_q - mean_p).square()) / logvar_p.exp() - 1)

class ConditionalVAE(USResidualUNet):
    def __init__(self):
        super().__init__(15, 16)
        self.prior = nn.Conv2d(32, 16, 1)
        self.target_encoder = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.SiLU())
        self.posterior = nn.Conv2d(48, 16, 1)
        self.latent_projection = nn.Conv2d(8, 32, 1)

    def forward(self, conditions, baseline_mm, target=None, mask=None, sample=False, prior_only=False,
                encoder_addition=None, use_latent=True, residual_latent=False, fine_addition=None, film=None):
        if conditions.shape[1:] != (4,15,6,6):
            raise ValueError('Expected four15-channel6x6 history frames')
        if not self.training and (target is not None or mask is not None):
            raise ValueError('Evaluation uses the conditional prior and cannot receive targets')
        encoded = self.enc(conditions.flatten(1,2))
        if encoder_addition is not None:
            if encoder_addition.shape != encoded.shape:
                raise ValueError('Encoder addition must match the6x6 encoded feature map')
            encoded = encoded + encoder_addition
        conditional = self.bottleneck(self.down(F.avg_pool2d(encoded,2)))
        mean_p, logvar_p = self.prior(conditional).chunk(2,dim=1)
        logvar_p = logvar_p.clamp(-8,8)
        baseline = F.interpolate(baseline_mm,scale_factor=4,mode='bilinear',align_corners=False)
        if self.training and not prior_only and use_latent:
            if target is None or mask is None:
                raise ValueError('Training posterior requires target and mask')
            residual = torch.where(mask,target-baseline,0)
            target_features = self.target_encoder(torch.cat([residual,mask.float()],dim=1))
            mean_q, logvar_q = self.posterior(torch.cat([conditional,target_features],dim=1)).chunk(2,dim=1)
            logvar_q = logvar_q.clamp(-8,8)
            latent = mean_q + torch.exp(.5*logvar_q)*torch.randn_like(mean_q)
            kl = gaussian_kl(mean_q,logvar_q,mean_p,logvar_p).mean()
        else:
            latent = mean_p
            if sample:
                latent = latent + torch.exp(.5*logvar_p)*torch.randn_like(latent)
            kl = conditional.new_zeros(())
        if residual_latent:
            result = self.decode_features(encoded,conditional,baseline,fine_addition,film)
            if use_latent and ((self.training and not prior_only) or sample):
                stochastic = self.decode_features(encoded,conditional+self.latent_projection(latent),baseline,fine_addition,film)
                centered = self.decode_features(encoded,conditional+self.latent_projection(mean_p),baseline,fine_addition,film)
                for key in ['field_mm','coarse_mm','fine_mm']:
                    result[key] = result[key]+stochastic[key]-centered[key]
        else:
            decoded = conditional+self.latent_projection(latent) if use_latent else conditional
            result = self.decode_features(encoded,decoded,baseline,fine_addition,film)
        result.update(kl=kl,prior_mean=mean_p,prior_logvar=logvar_p)
        return result

    def decode_features(self,encoded,decoded,baseline,fine_addition=None,film=None):
        hidden = self.dec(torch.cat([encoded,F.interpolate(decoded,size=(6,6),mode='bilinear',align_corners=False)],dim=1))
        if film is not None:
            hidden = hidden*(1+film[:,0,:,None,None])+film[:,1,:,None,None]
        coarse = self.coarse(hidden)
        if film is None:
            fine = self.fine(hidden)
        else:
            fine = self.fine[:3](hidden)
            fine = fine*(1+film[:,2,:,None,None])+film[:,3,:,None,None]
            fine = self.fine[3:](fine)
        if fine_addition is not None:
            fine = fine+fine_addition
        fine = fine-F.avg_pool2d(fine,4,4).repeat_interleave(4,-2).repeat_interleave(4,-1)
        field = baseline+F.interpolate(coarse,scale_factor=4,mode='nearest')+fine
        return dict(field_mm=field,baseline_mm=baseline,coarse_mm=coarse,fine_mm=fine)

def train_model(name, beta, data, train_indices, initial, device, align_prior=False):
    torch.manual_seed(42)
    model = USResidualUNet(15,16) if beta is None else ConditionalVAE()
    checkpoint_path = OUT / f'{name}.pt'
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert saved['beta']==beta and saved['epochs']==8 and saved['train_years']==[2008,2009]
        assert saved.get('align_prior',False)==align_prior
        torch.testing.assert_close(saved['mean'],data['mean'])
        torch.testing.assert_close(saved['std'],data['std'])
        model.load_state_dict(saved['model'])
        return model.to(device)
    model.load_state_dict(initial,strict=beta is None)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    rng = np.random.default_rng(42)
    factor = data['fine_height_factor'].float().to(device)[None,None]
    records = []
    started = time.time()
    for epoch in range(8):
        model.train()
        order = rng.permutation(train_indices[::6])
        errors = []
        kl_values = []
        for offset in range(0,len(order),16):
            current = order[offset:offset+16]
            indices = np.stack([current,current-1],axis=1).ravel()
            conditions,baseline,mask = batch(data,indices,device)
            if not mask.any():
                continue
            labels = data['target_h0'][indices].to(device)/20
            if beta is None:
                result = model(conditions,baseline)
                kl = labels.new_zeros(())
            else:
                result = model(conditions,baseline,target=labels,mask=mask)
                kl = result['kl']
            grid_loss = masked_mse(result['field_mm']*factor,labels*factor,mask)
            prediction,_,station_labels,station_mask = station_arrays(data,result,mask,indices,device)
            loss = grid_loss+.25*masked_mse(prediction,station_labels,station_mask)
            if align_prior:
                prior_result = model(conditions,baseline,prior_only=True)
                prior_grid_loss = masked_mse(prior_result['field_mm']*factor,labels*factor,mask)
                prior_prediction,_,prior_labels,prior_mask = station_arrays(data,prior_result,mask,indices,device)
                prior_loss = prior_grid_loss+.25*masked_mse(prior_prediction,prior_labels,prior_mask)
                loss = prior_loss+.25*loss
            if beta is not None:
                loss = loss+beta*min((epoch+1)/4,1)*kl
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CVAE loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            errors.append(float(grid_loss.detach())*400)
            kl_values.append(float(kl.detach()))
        record = dict(model=name,epoch=epoch+1,train_reconstruction_rmse_mm=float(np.sqrt(np.mean(errors))),
                      KL_per_latent=float(np.mean(kl_values)),seconds=time.time()-started)
        records.append(record)
        print(json.dumps(record),flush=True)
    pd.DataFrame(records).to_csv(OUT/f'{name}_training.csv',index=False)
    torch.save(dict(model=model.state_dict(),mean=data['mean'],std=data['std'],beta=beta,
                    train_years=[2008,2009],epochs=8,seed=42,align_prior=align_prior),OUT/f'{name}.pt')
    return model

def evaluate_samples(model,data,indices,held,truth,device):
    """Compute16-draw prior mean and latent-only interval diagnostics on common masks."""
    model.eval()
    torch.manual_seed(142)
    coordinates = torch.tensor(held[['x_km','y_km']].to_numpy()/300,dtype=torch.float32,device=device)[None]
    station_factor = torch.tensor(np.exp(-held.orthometric_height_m.to_numpy()/3000),dtype=torch.float32,device=device)[None]
    grid_factor = data['fine_height_factor'].float().to(device)[None,None]
    grid_sse = station_sse = spread_sum = 0.
    grid_count = station_count = covered = 0
    with torch.inference_mode():
        for offset in range(0,len(indices),128):
            times = indices[offset:offset+128]
            conditions,baseline,mask = batch(data,times,device)
            draws = torch.stack([model(conditions,baseline,sample=True)['field_mm'] for _ in range(16)])
            mean = draws.mean(0)
            labels = data['target_h0'][times].to(device)/20
            grid_sse += float(((mean-labels)*grid_factor*20).square()[mask].sum())
            grid_count += int(mask.sum())
            query = coordinates.expand(len(times),-1,-1)
            station_draws = torch.stack([sample_points(draw,query)*station_factor*20 for draw in draws])
            target = torch.tensor(truth[times],device=device)
            valid = torch.isfinite(target)&(sample_points(mask.float(),query)>=1-1e-6)
            station_sse += float((station_draws.mean(0)-target).square()[valid].sum())
            station_count += int(valid.sum())
            lower,upper = torch.quantile(station_draws,torch.tensor([.05,.95],device=device),dim=0)
            covered += int(((target>=lower)&(target<=upper)&valid).sum())
            spread_sum += float(station_draws.std(0,unbiased=False)[valid].sum())
    return dict(grid_rmse_mm=float(np.sqrt(grid_sse/grid_count)),station_rmse_mm=float(np.sqrt(station_sse/station_count)),
                grid_points=grid_count,station_records=station_count,samples=16,
                mean_station_latent_std_mm=spread_sum/station_count,
                empirical90percent_interval_coverage=covered/station_count,
                interval_scope='latent sampling only; excludes observation/reference and parameter uncertainty')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data,train,_,years,hours = load(train_years=(2008,2009))
    development = valid_indices(years,[2010])
    held,truth = development_truth(hours,years)
    torch.manual_seed(42)
    initial = {key:value.clone() for key,value in USResidualUNet(15,16).state_dict().items()}
    existing = pd.read_csv(STUDY/'04_models/feature_learning_pilot_v1/four_arm_comparison.csv')
    gan = existing[existing.arm.eq('baseline')].iloc[0].to_dict()
    records = [dict(model='matched_GAN',inference='deterministic',grid_rmse_mm=gan['grid_rmse_mm'],
                    station_rmse_mm=gan['station_rmse_mm'],grid_points=int(gan['grid_points']),station_records=int(gan['station_records']))]
    for name,beta,aligned in [('UNet_noGAN',None,False),('CVAE_beta001',.001,False),
                              ('CVAE_beta01',.01,False),('CVAE_prior_aligned',.01,True)]:
        model = train_model(name,beta,data,train,initial,device,align_prior=aligned)
        metric = evaluate(model,data,development,held,truth,device)
        metric.update(model=name,inference='deterministic' if beta is None else 'prior_mean',parameters=sum(p.numel() for p in model.parameters()))
        records.append(metric)
        print('DEVELOPMENT',json.dumps(metric),flush=True)
        if beta is not None:
            metric = evaluate_samples(model,data,development,held,truth,device)
            metric.update(model=name,inference='prior_MC16',parameters=sum(p.numel() for p in model.parameters()))
            records.append(metric)
            print('MONTE_CARLO',json.dumps(metric),flush=True)
    comparison = pd.DataFrame(records)
    assert comparison.grid_points.nunique()==comparison.station_records.nunique()==1
    comparison.to_csv(OUT/'comparison.csv',index=False)
    result = dict(status='complete',train_years=[2008,2009],development_year=2010,
                  epochs=8,seed=42,training_stride_hours=6,prior_only_evaluation=True)
    result.update(used_2011=False,independent_test_observations_read=False,results=records)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('CVAE PILOT COMPLETE',flush=True)

if __name__=='__main__':
    main()
