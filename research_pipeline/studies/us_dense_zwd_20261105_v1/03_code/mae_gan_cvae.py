"""MAE-GAN-CVAE fusion and matched raw-station-information ablations."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from cvae_pilot import ConditionalVAE
from station_mae import StationArchive, StationMAE, pretrain
from train_us_gan import load, batch, station_arrays, masked_mse
from feature_learning import development_truth
from us_gan import sample_points, valid_indices, critic_loss
from build_reference_grid import weights

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY/'04_models/mae_gan_cvae_v2'

class FusionGenerator(nn.Module):
    def __init__(self,use_latent=True,residual_latent=False):
        super().__init__()
        self.core = ConditionalVAE()
        self.mae = StationMAE()
        self.projection = nn.Conv2d(48,16,1)
        self.use_latent = use_latent
        self.residual_latent = residual_latent

    def token_features(self,tokens,padding,queries):
        context = self.mae.representations(tokens,padding,queries)
        return self.projection(context.transpose(1,2).reshape(-1,48,6,6))

    def forward(self,conditions,baseline,tokens=None,padding=None,queries=None,addition=None,
                target=None,mask=None,prior_only=True,sample=False):
        if addition is None:
            addition = self.token_features(tokens,padding,queries)
        return self.core(conditions,baseline,target=target,mask=mask,prior_only=prior_only,
                         sample=sample,encoder_addition=addition,use_latent=self.use_latent,residual_latent=self.residual_latent)

class SpatialCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(spectral_norm(nn.Conv2d(9,16,3)),nn.LeakyReLU(.2),
                                 spectral_norm(nn.Conv2d(16,16,1)),nn.LeakyReLU(.2),
                                 spectral_norm(nn.Conv2d(16,1,1)))

    def forward(self,residual,condition,mask):
        clean = torch.where(mask,residual,0)
        score = self.net(torch.cat([clean,condition],dim=1))
        valid = F.avg_pool2d(mask.float(),3,stride=1)>=1-1e-6
        return score[valid]

class TemporalCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(spectral_norm(nn.Conv1d(4,16,2)),nn.LeakyReLU(.2),
                                 spectral_norm(nn.Conv1d(16,16,1)),nn.LeakyReLU(.2),
                                 spectral_norm(nn.Conv1d(16,1,1)))

    def forward(self,pairs):
        return self.net(pairs).flatten()

def token_inputs(archive,data,indices,hours,device):
    archive_indices = hours[indices]-archive.hours[0]
    visible = np.broadcast_to(archive.sparse[None],(len(indices),39))
    tokens,padding = archive.inputs(archive_indices,visible,device)
    raw = data['conditions'][0]
    positions = np.c_[raw[7].numpy().ravel()*.75,raw[8].numpy().ravel()*.75,raw[6].numpy().ravel()/3000].astype('f4')
    queries = archive.queries(archive_indices,positions,device)
    return tokens,padding,queries

def spatial_condition(normalized,factor):
    low = normalized[:,0,[0,1,2,3,4,13,14]]
    condition = F.interpolate(low,scale_factor=4,mode='bilinear',align_corners=False)
    height = -torch.log(factor.clamp_min(1e-6)).expand(len(low),-1,-1,-1)
    return torch.cat([condition,height],dim=1)

def temporal_pairs(residual,normalized,station_xy,valid):
    query = station_xy.float().to(residual.device).clamp(-1,1)[None].expand(len(residual),-1,-1)
    weather = F.grid_sample(normalized[:,0,[1,2,4]],query.unsqueeze(2),align_corners=False,padding_mode='border')[:,:,:,0].transpose(1,2)
    values = torch.cat([residual[:,:,None],weather],dim=2)
    values = values.reshape(-1,2,values.shape[1],4).permute(0,2,3,1)
    pair_mask = valid.reshape(-1,2,valid.shape[1]).all(1)
    return values[pair_mask]

def train_arm(name,mae_enabled,gan_enabled,latent_enabled,archive,data,train,hours,initial,pretrained,device,residual_latent=False,
              model_factory=FusionGenerator,input_builder=token_inputs,epochs=8,seed=42,output_dir=None,supervised_years=(2008,2009),random_hour_subset=False):
    output_dir = OUT if output_dir is None else output_dir
    torch.manual_seed(seed)
    model = model_factory(latent_enabled,residual_latent).to(device)
    model.load_state_dict(initial)
    if mae_enabled:
        model.mae.load_state_dict(pretrained)
    optimizer = torch.optim.AdamW([
        {'params':model.core.parameters(),'lr':3e-4},
        {'params':model.projection.parameters(),'lr':3e-4},
        {'params':model.mae.parameters(),'lr':1e-4}])
    spatial = SpatialCritic().to(device)
    temporal = TemporalCritic().to(device)
    optimizers = [torch.optim.AdamW(spatial.parameters(),lr=1e-4),torch.optim.AdamW(temporal.parameters(),lr=1e-4)]
    rng = np.random.default_rng(seed)
    factor = data['fine_height_factor'].float().to(device)[None,None]
    records = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train)[:len(train[::6])] if random_hour_subset else rng.permutation(train[::6])
        errors = []
        kl_values = []
        patch_count = pair_count = 0
        for offset in range(0,len(order),16):
            current = order[offset:offset+16]
            indices = np.stack([current,current-1],axis=1).ravel()
            x,base,mask = batch(data,indices,device)
            if not mask.any():
                continue
            tokens,padding,queries = input_builder(archive,data,indices,hours,device)
            addition = model.token_features(tokens,padding,queries)
            result = model(x,base,addition=addition,prior_only=True)
            label = data['target_h0'][indices].to(device)/20
            grid_loss = masked_mse(result['field_mm']*factor,label*factor,mask)
            sp,sb,sl,sm = station_arrays(data,result,mask,indices,device)
            loss = grid_loss+.25*masked_mse(sp,sl,sm)
            kl = label.new_zeros(())
            if latent_enabled:
                posterior = model(x,base,addition=addition,target=label,mask=mask,prior_only=False)
                pp,_,pl,pm = station_arrays(data,posterior,mask,indices,device)
                posterior_loss = masked_mse(posterior['field_mm']*factor,label*factor,mask)+.25*masked_mse(pp,pl,pm)
                kl = posterior['kl']
                loss = loss+.25*posterior_loss+.01*min((epoch+1)/4,1)*kl
            if gan_enabled and epoch>=2:
                condition = spatial_condition(x,factor)
                fake_field = (result['field_mm']-result['baseline_mm'])*factor
                real_field = (label-result['baseline_mm'])*factor
                fake_pairs = temporal_pairs(sp-sb,x,data['station_xy'],sm)
                real_pairs = temporal_pairs(sl-sb,x,data['station_xy'],sm)
                for critic in [spatial,temporal]:
                    critic.train()
                    for parameter in critic.parameters():
                        parameter.requires_grad_(True)
                real_scores = spatial(real_field.detach(),condition,mask)
                fake_scores = spatial(fake_field.detach(),condition,mask)
                if real_scores.numel():
                    optimizers[0].zero_grad(set_to_none=True)
                    critic_loss(real_scores,fake_scores).backward()
                    optimizers[0].step()
                    spatial.eval()
                    for parameter in spatial.parameters():parameter.requires_grad_(False)
                    loss = loss-.005*spatial(fake_field,condition,mask).mean()
                    patch_count += len(real_scores)
                if len(real_pairs):
                    optimizers[1].zero_grad(set_to_none=True)
                    critic_loss(temporal(real_pairs.detach()),temporal(fake_pairs.detach())).backward()
                    optimizers[1].step()
                    temporal.eval()
                    for parameter in temporal.parameters():parameter.requires_grad_(False)
                    loss = loss-.005*temporal(fake_pairs).mean()
                    pair_count += len(real_pairs)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fusion loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            errors.append(float(grid_loss.detach())*400)
            kl_values.append(float(kl.detach()))
        record = dict(arm=name,epoch=epoch+1,train_prior_grid_rmse_mm=float(np.sqrt(np.mean(errors))),
                      KL_per_latent=float(np.mean(kl_values)),spatial_valid_patches=patch_count,temporal_pairs=pair_count,seconds=time.time()-started)
        records.append(record)
        pd.DataFrame(records).to_csv(output_dir/f'{name}_training.csv',index=False)
        print(json.dumps(record),flush=True)
        if gan_enabled and epoch>=2 and (patch_count==0 or pair_count==0):
            raise ValueError('A required discriminator received no valid training examples')
    torch.save(dict(model=model.state_dict(),spatial=spatial.state_dict(),temporal=temporal.state_dict(),
                    mean=data['mean'],std=data['std'],token_mean=archive.mean,token_std=archive.std,
                    MAE=mae_enabled,GAN=gan_enabled,CVAE=latent_enabled,residual_latent=residual_latent,supervised_years=list(supervised_years),pretrain_years=list(range(2000,2010)) if mae_enabled else [],epochs=epochs,seed=seed,random_hour_subset=random_hour_subset),output_dir/f'{name}.pt')
    return model

def evaluate_model(model,archive,data,indices,hours,held,truth,device):
    model.eval()
    xy = torch.tensor(held[['x_km','y_km']].to_numpy()/300,dtype=torch.float32,device=device)[None]
    sf = torch.tensor(np.exp(-held.orthometric_height_m.to_numpy()/3000),dtype=torch.float32,device=device)[None]
    factor = data['fine_height_factor'].float().to(device)[None,None]
    squared = station_squared = 0.
    count = station_count = 0
    with torch.inference_mode():
        for offset in range(0,len(indices),64):
            times = indices[offset:offset+64]
            x,base,mask = batch(data,times,device)
            tokens,padding,queries = token_inputs(archive,data,times,hours,device)
            prediction = model(x,base,tokens,padding,queries)['field_mm']
            label = data['target_h0'][times].to(device)/20
            squared += float(((prediction-label)*factor*20).square()[mask].sum())
            count += int(mask.sum())
            query = xy.expand(len(times),-1,-1)
            station = sample_points(prediction,query)*sf*20
            target = torch.tensor(truth[times],device=device)
            valid = torch.isfinite(target)&(sample_points(mask.float(),query)>=1-1e-6)
            station_squared += float((station-target).square()[valid].sum())
            station_count += int(valid.sum())
    return dict(grid_rmse_mm=float(np.sqrt(squared/count)),station_rmse_mm=float(np.sqrt(station_squared/station_count)),
                grid_points=count,station_records=station_count)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    archive = StationArchive()
    pretrained = pretrain(archive,STUDY/'04_models/mae_gan_cvae_v1',device)
    data,train,_,years,hours = load(train_years=(2008,2009))
    development = valid_indices(years,[2010])
    held,truth = development_truth(hours,years)
    torch.manual_seed(42)
    initial = {key:value.clone() for key,value in FusionGenerator().state_dict().items()}
    arms = [('MAE_GAN_CVAE',True,True,True),('random_GAN_CVAE',False,True,True),
            ('MAE_CVAE',True,False,True),('MAE_GAN',True,True,False),
            ('MAE_deterministic',True,False,False),('random_deterministic',False,False,False)]
    results = []
    for name,mae,gan,latent in arms:
        path = OUT/f'{name}.pt'
        if path.exists():
            saved = torch.load(path,map_location='cpu',weights_only=False)
            assert saved['MAE']==mae and saved['GAN']==gan and saved['CVAE']==latent
            model = FusionGenerator(latent).to(device)
            model.load_state_dict(saved['model'])
        else:
            model = train_arm(name,mae,gan,latent,archive,data,train,hours,initial,pretrained,device)
        metrics = evaluate_model(model,archive,data,development,hours,held,truth,device)
        metrics.update(arm=name,MAE=mae,GAN=gan,CVAE=latent,parameters=sum(p.numel() for p in model.parameters()))
        results.append(metrics)
        pd.DataFrame(results).to_csv(OUT/'development_comparison.csv',index=False)
        print('DEVELOPMENT',json.dumps(metrics),flush=True)
    table = pd.DataFrame(results)
    assert table.grid_points.nunique()==table.station_records.nunique()==1
    summary = dict(status='complete',primary_model='MAE_GAN_CVAE',pretrain_years=list(range(2000,2010)),
                   supervised_years=[2008,2009],development_year=2010,seed=42,epochs=8,
                   comparison_information='same raw39station histories and coarse fields',independent_test_observations_read=False,results=results)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
