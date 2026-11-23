"""Registered development selection and locked final spatial evaluation."""
from pathlib import Path
import argparse
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from regional_temporal_mae import RegionalGenerator, pretrain, prepare_cache, file_sha256
from moment_superresolution import MomentFeatures, inputs, evaluate, validation_truth
from mae_gan_cvae import SpatialCritic, TemporalCritic, spatial_condition, temporal_pairs
from train_us_gan import load, batch, station_arrays, masked_mse
from feature_learning import development_truth
from us_gan import valid_indices, critic_loss

STUDY=Path(__file__).resolve().parents[1]
OUT=STUDY/'04_models/architecture_search_v1'


class StationCritic(nn.Module):
    def __init__(self,xy,height,graph=False):
        super().__init__()
        self.graph=graph
        self.node=nn.Sequential(spectral_norm(nn.Linear(8,32)),nn.LeakyReLU(.2),
                                spectral_norm(nn.Linear(32,32)),nn.LeakyReLU(.2))
        self.readout=spectral_norm(nn.Linear(32,1))
        distance=torch.cdist(xy,xy)*300
        source,destination=torch.where((distance>0)&(distance<=100))
        geometry=torch.cat([(xy[source]-xy[destination])*3,
                            (height[source]-height[destination])[:,None],
                            distance[source,destination,None]/100],dim=1)
        self.register_buffer('source',source,persistent=False)
        self.register_buffer('destination',destination,persistent=False)
        self.register_buffer('geometry',geometry,persistent=False)
        if graph:
            self.messages=nn.ModuleList([nn.Sequential(spectral_norm(nn.Linear(68,32)),nn.LeakyReLU(.2),
                                                      spectral_norm(nn.Linear(32,32)),nn.LeakyReLU(.2)) for _ in range(2)])
            self.updates=nn.ModuleList([nn.Sequential(spectral_norm(nn.Linear(64,32)),nn.LeakyReLU(.2)) for _ in range(2)])

    def forward(self,residual,condition,mask):
        values=torch.cat([residual[:,:,None],condition],dim=-1).masked_fill(~mask[:,:,None],0)
        hidden=self.node(values)*mask[:,:,None]
        if self.graph:
            edge_mask=mask[:,self.source]&mask[:,self.destination]
            geometry=self.geometry[None].expand(len(values),-1,-1)
            for message,update in zip(self.messages,self.updates):
                edge=torch.cat([hidden[:,self.destination],hidden[:,self.source]-hidden[:,self.destination],geometry],dim=-1)
                messages=message(edge)*edge_mask[:,:,None]
                aggregate=torch.zeros_like(hidden).index_add(1,self.destination,messages)
                degree=hidden.new_zeros((*mask.shape,1)).index_add(1,self.destination,edge_mask[:,:,None].float())
                hidden=update(torch.cat([hidden,aggregate/degree.clamp_min(1)],dim=-1))*mask[:,:,None]
        pooled=hidden.sum(1)/mask.sum(1).clamp_min(1)[:,None]
        return self.readout(pooled).squeeze(-1)


class SearchGenerator(RegionalGenerator):
    def __init__(self,variant='base',use_latent=False,fine_height=None):
        super().__init__(use_latent)
        self.variant=variant
        if 'film' in variant:
            self.film_head=nn.Linear(64,64)
            nn.init.zeros_(self.film_head.weight)
            nn.init.zeros_(self.film_head.bias)
        if 'query' in variant:
            self.query_head=nn.Linear(48,1)
            nn.init.zeros_(self.query_head.weight)
            nn.init.zeros_(self.query_head.bias)
            self.query_gate=nn.Parameter(torch.tensor(.05))
            axis=torch.arange(24,dtype=torch.float32)*25-287.5
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            height=torch.zeros(576) if fine_height is None else fine_height.reshape(-1).float()/3000
            self.register_buffer('fine_positions',torch.stack([xx.reshape(-1)/400,yy.reshape(-1)/400,height],dim=1),persistent=False)

    def forward(self,conditions,baseline,tokens,padding,queries,target=None,mask=None,prior_only=True,sample=False):
        if self.variant=='no_history':
            return self.core(conditions,baseline,target=target,mask=mask,prior_only=prior_only,
                             sample=sample,use_latent=self.use_latent)
        tokens=tokens.masked_fill(padding[:,:,None],0)
        addition=self.token_features(tokens,padding,queries)
        film=None
        fine_addition=None
        if 'film' in self.variant:
            valid=(~padding).float()
            pooled=(tokens[:,:,:64]*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
            film=.1*torch.tanh(self.film_head(pooled).reshape(-1,4,16))
        if 'query' in self.variant:
            q=torch.cat([self.fine_positions[None].expand(len(tokens),-1,-1),queries[:,0:1,3:].expand(-1,576,-1)],dim=-1)
            encoded=self.mae.embedding(tokens[:,:,:64])+self.mae.position(tokens[:,:,64:])
            query=self.mae.query(q)
            distance=torch.cdist(q[:,:,:2],tokens[:,:,64:66])/.25
            attention_mask=(-distance).masked_fill(padding[:,None,:],float('-inf')).repeat_interleave(4,dim=0)
            context,_=self.mae.attention(query,encoded,encoded,attn_mask=attention_mask,need_weights=False)
            context=self.mae.norm(query+context)
            fine_addition=torch.tanh(self.query_gate)*self.query_head(context).transpose(1,2).reshape(-1,1,24,24)
        return self.core(conditions,baseline,target=target,mask=mask,prior_only=prior_only,sample=sample,
                         encoder_addition=addition,use_latent=self.use_latent,
                         fine_addition=fine_addition,film=film)


def station_conditions(normalized,data):
    xy=data['station_xy'].float().to(normalized.device)
    query=xy.clamp(-1,1)[None,:,None].expand(len(normalized),-1,-1,-1)
    weather=F.grid_sample(normalized[:,0,[1,2,3,4]],query,align_corners=False,padding_mode='border')[:,:,:,0].transpose(1,2)
    height=-torch.log(data['station_height_factor'].float().to(normalized.device))
    geometry=torch.cat([xy,height[:,None]],dim=1)[None].expand(len(normalized),-1,-1)
    return torch.cat([weather,geometry],dim=2)


def identifier(config,seed):
    return f"{config['generator']}_{config['critic']}_cvae{int(config['latent'])}_seed{seed}"


def make_model(config,data):
    height=-3000*torch.log(data['fine_height_factor'])
    return SearchGenerator(config['generator'],config['latent'],height).cuda()


def state_fingerprint(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_shared(model,initial_state):
    state=model.state_dict()
    for key,value in state.items():
        if key not in initial_state or initial_state[key].shape!=value.shape:
            raise ValueError(f'Missing or incompatible canonical parameter: {key}')
    model.load_state_dict({key:initial_state[key] for key in state})


def station_difference_loss(prediction,target,mask,source,destination):
    residual=torch.where(mask,prediction-target,0)
    differences=residual[:,source]-residual[:,destination]
    supported=mask[:,source]&mask[:,destination]
    return masked_mse(differences,torch.zeros_like(differences),supported)


def train(config,seed,archive,data,train_indices,hours,directory,lineage,epochs=20,warmup_epochs=5,epoch_orders=None,
          initial_state=None,stage_callback=None):
    name=identifier(config,seed)
    path=directory/f'{name}.pt'
    order_sha=hashlib.sha256(np.concatenate(epoch_orders).tobytes()).hexdigest() if epoch_orders is not None else None
    contract=dict(epochs=epochs,warmup_epochs=warmup_epochs,order_sha256=order_sha,seed=seed)
    if initial_state is not None:
        contract['initial_sha256']=state_fingerprint(initial_state)
    torch.manual_seed(seed)
    model=make_model(config,data)
    if initial_state is not None:
        initialize_shared(model,initial_state)
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['config']==config and saved['lineage']==lineage and saved['epochs']==epochs
        assert saved.get('warmup_epochs',5)==warmup_epochs
        if epoch_orders is not None:
            assert saved['training_contract']==contract
        torch.testing.assert_close(saved['mean'],data['mean'])
        torch.testing.assert_close(saved['std'],data['std'])
        model.load_state_dict(saved['model'])
        return model
    core_ids={id(p) for p in model.core.parameters()}
    optimizer=torch.optim.AdamW([{'params':model.core.parameters(),'lr':3e-4},
        {'params':[p for p in model.parameters() if id(p) not in core_ids],'lr':1e-4}])
    warm_path=directory/f"warmup_{config['generator']}_cvae{int(config['latent'])}_seed{seed}.pt"
    records=[]
    first_epoch=0
    if warm_path.exists():
        warm=torch.load(warm_path,map_location='cpu',weights_only=False)
        assert warm['lineage']==lineage
        assert len(warm['records'])==warmup_epochs
        if epoch_orders is not None:
            assert warm['training_contract']==contract
        model.load_state_dict(warm['model'])
        optimizer.load_state_dict(warm['optimizer'])
        records=warm['records']
        first_epoch=warmup_epochs
    torch.manual_seed(seed+1000)
    critics={}
    kind=config['critic']
    if 'patch' in kind:
        critics['patch']=SpatialCritic().cuda()
    if kind in ['set','graph','graph_time']:
        critics['station']=StationCritic(data['station_xy'].float(),-torch.log(data['station_height_factor'].float()),graph=kind!='set').cuda()
    if 'time' in kind:
        critics['time']=TemporalCritic().cuda()
    d_optimizers={key:torch.optim.AdamW(value.parameters(),lr=1e-4) for key,value in critics.items()}
    factor=data['fine_height_factor'].float().cuda()[None,None]
    if config.get('spatial_weight',0):
        distances=torch.cdist(data['station_xy'].float(),data['station_xy'].float())*300
        spatial_source,spatial_destination=torch.where(torch.triu((distances>0)&(distances<=100),diagonal=1))
        spatial_source,spatial_destination=spatial_source.cuda(),spatial_destination.cuda()
    started=time.time()
    for epoch in range(first_epoch,epochs):
        torch.manual_seed(seed+10000*epoch)
        order=epoch_orders[epoch] if epoch_orders is not None else np.random.default_rng(seed+10000*epoch).permutation(train_indices)[:len(train_indices[::6])]
        model.train()
        squared=count=0
        kl_total=0.
        kl_batches=0
        totals={key:0 for key in critics}
        ratio=cosine=None
        first_batch_scores={}
        for offset in range(0,len(order),16):
            current=order[offset:offset+16]
            indices=np.stack([current,current-1],axis=1).ravel()
            x,base,mask=batch(data,indices,'cuda')
            tokens,padding,queries=inputs(archive,data,indices,hours,'cuda')
            result=model(x,base,tokens,padding,queries)
            prediction=result['field_mm']
            label=data['target_h0'][indices].cuda()/20
            grid_loss=masked_mse(prediction*factor,label*factor,mask)
            sp,sb,sl,sm=station_arrays(data,result,mask,indices,'cuda')
            supervised=grid_loss+.25*masked_mse(sp,sl,sm)
            if config.get('spatial_weight',0):
                supervised=supervised+config['spatial_weight']*station_difference_loss(sp,sl,sm,spatial_source,spatial_destination)
            if config.get('temporal_weight',0):
                supervised=supervised+config['temporal_weight']*masked_mse(sp[::2]-sp[1::2],sl[::2]-sl[1::2],sm[::2]&sm[1::2])
            loss=supervised
            kl=prediction.new_zeros(())
            if config['latent']:
                posterior=model(x,base,tokens,padding,queries,target=label,mask=mask,prior_only=False)
                pp,_,pl,pm=station_arrays(data,posterior,mask,indices,'cuda')
                kl=posterior['kl']
                kl_total+=float(kl.detach())
                kl_batches+=1
                loss=loss+.25*(masked_mse(posterior['field_mm']*factor,label*factor,mask)+.25*masked_mse(pp,pl,pm))+.01*min((epoch+1)/4,1)*kl
            adversarial=prediction.new_zeros(())
            if epoch>=warmup_epochs:
                for key,critic in critics.items():
                    critic.train().requires_grad_(True)
                    if key=='patch':
                        condition=spatial_condition(x,factor)
                        fake=(prediction-result['baseline_mm'])*factor
                        real=(label-result['baseline_mm'])*factor
                        args=(condition,mask)
                    elif key=='station':
                        available=sm.any(1)
                        fake=(sp-sb)[available]
                        real=(sl-sb)[available]
                        args=(station_conditions(x,data)[available],sm[available])
                    else:
                        fake=temporal_pairs(sp-sb,x,data['station_xy'],sm)
                        real=temporal_pairs(sl-sb,x,data['station_xy'],sm)
                        args=()
                    real_score=critic(real.detach(),*args)
                    fake_score=critic(fake.detach(),*args)
                    if real_score.numel()==0:
                        continue
                    if offset==0:
                        first_batch_scores[key]=dict(real=float(real_score.detach().mean()),fake=float(fake_score.detach().mean()))
                        if key=='station':
                            first_batch_scores[key]['mean_nodes']=float(sm.float().sum(1).mean())
                            if critic.graph:
                                first_batch_scores[key]['mean_directed_edges']=float((sm[:,critic.source]&sm[:,critic.destination]).float().sum(1).mean())
                    d_optimizers[key].zero_grad(set_to_none=True)
                    critic_loss(real_score,fake_score).backward()
                    d_optimizers[key].step()
                    critic.eval().requires_grad_(False)
                    adversarial=adversarial-.005*critic(fake,*args).mean()
                    totals[key]+=real_score.numel()
                if critics and offset==0 and adversarial.requires_grad:
                    parameter=model.core.enc.net[0].weight
                    g_supervised=torch.autograd.grad(supervised,parameter,retain_graph=True)[0].flatten()
                    g_adversarial=torch.autograd.grad(adversarial,parameter,retain_graph=True)[0].flatten()
                    ratio=float(g_adversarial.norm()/g_supervised.norm().clamp_min(1e-12))
                    cosine=float(F.cosine_similarity(g_supervised[None],g_adversarial[None]))
            loss=loss+adversarial
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            squared+=float(((prediction.detach()-label)*factor*20).square()[mask].sum())
            count+=int(mask.sum())
        record=dict(run=name,epoch=epoch+1,train_grid_rmse_mm=np.sqrt(squared/count),
                    mean_kl_per_latent=kl_total/max(kl_batches,1),
                    discriminator_examples=totals,first_batch_gradient_ratio=ratio,first_batch_gradient_cosine=cosine,
                    first_batch_scores=first_batch_scores,
                    seconds_since_resume=time.time()-started)
        records.append(record)
        (directory/f'{name}_training.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print('TRAIN',json.dumps(record),flush=True)
        if stage_callback is not None:
            stage_callback(model,epoch+1,record)
        if epoch==warmup_epochs-1:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),records=records.copy(),lineage=lineage,training_contract=contract),warm_path)
    torch.save(dict(model=model.state_dict(),critics={key:value.state_dict() for key,value in critics.items()},
        config=config,seed=seed,epochs=epochs,warmup_epochs=warmup_epochs,training_contract=contract,lineage=lineage,mean=data['mean'],std=data['std']),path)
    return model


def score(config,seed,model,archive,data,indices,hours,held,truth,directory):
    name=identifier(config,seed)
    metrics=evaluate(model,archive,data,indices,hours,held,truth,name,output_dir=directory)
    metrics.update(generator=config['generator'],critic=config['critic'],latent=config['latent'],seed=seed,
                   parameters=sum(p.numel() for p in model.parameters()))
    (directory/f'{name}_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('SCORE',json.dumps(metrics),flush=True)
    return metrics


def config_from(row):
    return dict(generator=row['generator'],critic=row['critic'],latent=bool(row['latent']))


def prepare_development():
    encoder_dir=OUT/'development_encoder'
    encoder_dir.mkdir(parents=True,exist_ok=True)
    cache_path=STUDY/'02_data/processed/architecture_search_mae_cutoff2009.h5'
    state=pretrain(encoder_dir,last_year=2009)
    assert max(state['years'])==2009
    prepare_cache(state,encoder_dir,cache_path)
    return MomentFeatures(cache_path),dict(encoder_sha256=file_sha256(encoder_dir/'regional_mae.pt'),
        cache_sha256=file_sha256(cache_path),pretrain_cutoff=2009,supervised_years=[2008,2009])


def final_truth(hours,years,role):
    if role=='validation':
        return validation_truth(hours,years)
    roles=pd.read_csv(STUDY/'02_data/processed/reference_25km_v1/station_roles.csv')
    held=roles[roles.in_main_2008_2010 & roles.role.eq('independent_test') & roles.in_core].copy()
    truth=np.full((len(hours),len(held)),np.nan,dtype='f4')
    first=int(hours[np.flatnonzero(years==2011)[0]])
    for j,row in enumerate(held.itertuples()):
        frame=pd.read_parquet(STUDY/f'02_data/processed/pwv_local/period_2007_2011/by_station/{row.station}.parquet',
                              columns=['hour_id','zwd_mm','joint_valid'])
        frame=frame[frame.joint_valid & frame.hour_id.between(first,int(hours[-1]))]
        truth[frame.hour_id.to_numpy()-hours[0],j]=frame.zwd_mm.to_numpy()
    return held,truth


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--screen-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():
        print('Experiment already complete; inspect saved results.',flush=True)
        return
    dev=OUT/'development'
    dev.mkdir(exist_ok=True)
    archive,lineage=prepare_development()
    data,train_indices,_,years,hours=load(train_years=(2008,2009))
    indices=valid_indices(years,[2010])
    held,truth=development_truth(hours,years)
    screened=[]
    for critic in ['none','patch','set','graph','graph_time','patch_time']:
        config=dict(generator='base',critic=critic,latent=False)
        model=train(config,42,archive,data,train_indices,hours,dev,lineage)
        screened.append(score(config,42,model,archive,data,indices,hours,held,truth,dev))
        pd.DataFrame(screened).to_csv(dev/'screening.csv',index=False)
    best_critic=min(screened,key=lambda row:row['station_macro_rmse_mm'])['critic']
    for variant in ['film','query','query_film']:
        config=dict(generator=variant,critic=best_critic,latent=False)
        model=train(config,42,archive,data,train_indices,hours,dev,lineage)
        screened.append(score(config,42,model,archive,data,indices,hours,held,truth,dev))
        pd.DataFrame(screened).to_csv(dev/'screening.csv',index=False)
    best=config_from(min(screened,key=lambda row:row['station_macro_rmse_mm']))
    adversarial_critic=min([row for row in screened if row['critic']!='none'],key=lambda row:row['station_macro_rmse_mm'])['critic']
    fusion=dict(generator=best['generator'],critic=adversarial_critic,latent=False)
    if fusion not in [config_from(row) for row in screened]:
        model=train(fusion,42,archive,data,train_indices,hours,dev,lineage)
        screened.append(score(fusion,42,model,archive,data,indices,hours,held,truth,dev))
    latent_candidates=[{**best,'latent':True}]
    fusion['latent']=True
    if fusion not in latent_candidates:
        latent_candidates.append(fusion)
    for config in latent_candidates:
        model=train(config,42,archive,data,train_indices,hours,dev,lineage)
        screened.append(score(config,42,model,archive,data,indices,hours,held,truth,dev))
    pd.DataFrame(screened).to_csv(dev/'screening.csv',index=False)
    ranked=sorted(screened,key=lambda row:(row['station_macro_rmse_mm'],row['parameters']))
    candidates=[config_from(row) for row in ranked[:2]]
    baseline=dict(generator='base',critic='none',latent=False)
    if baseline not in candidates:
        candidates.append(baseline)
    best_fusion=config_from(min([row for row in screened if row['latent'] and row['critic']!='none'],key=lambda row:row['station_macro_rmse_mm']))
    if best_fusion not in candidates:
        candidates.append(best_fusion)
    confirmed=[]
    for config in candidates:
        for seed in [42,17,101]:
            model=train(config,seed,archive,data,train_indices,hours,dev,lineage)
            confirmed.append(score(config,seed,model,archive,data,indices,hours,held,truth,dev))
            pd.DataFrame(confirmed).to_csv(dev/'confirmation.csv',index=False)
    table=pd.DataFrame(confirmed)
    summary=table.groupby(['generator','critic','latent'],as_index=False).agg(
        macro_rmse=('station_macro_rmse_mm','mean'),macro_sd=('station_macro_rmse_mm','std'),
        micro_rmse=('station_rmse_mm','mean'),parameters=('parameters','first'))
    summary.to_csv(dev/'confirmation_summary.csv',index=False)
    selected=config_from(summary.sort_values(['macro_rmse','parameters']).iloc[0])
    selection=dict(selected=selected,best_full_fusion=best_fusion,rule='lowest 3seed development macroRMSE',development_year=2010,
        independent_test_read=False,candidate_count=len(screened),confirmed_configs=candidates,
        protocol_sha256=file_sha256(STUDY/'00_protocol/architecture_search_v1.json'))
    selection_path=OUT/'locked_selection.json'
    if selection_path.exists():
        assert json.loads(selection_path.read_text(encoding='utf-8'))==selection
    else:
        selection_path.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('LOCKED',json.dumps(selection),flush=True)
    if args.screen_only:
        (OUT/'screen_complete.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        return
    final_dir=OUT/'final_2008_2010'
    final_dir.mkdir(exist_ok=True)
    final_cache=STUDY/'02_data/processed/regional_mae_512h_6h_features.h5'
    archive=MomentFeatures(final_cache)
    data,train_indices,val,years,hours=load(train_years=(2008,2009,2010))
    lineage=dict(encoder_sha256=file_sha256(STUDY/'04_models/regional_temporal_mae_v1/regional_mae.pt'),
                 cache_sha256=file_sha256(final_cache),pretrain_cutoff=2010,supervised_years=[2008,2009,2010])
    configs=[baseline] if selected==baseline else [baseline,selected]
    if best_fusion not in configs:
        configs.append(best_fusion)
    for config in configs:
        for seed in [42,17,101]:
            train(config,seed,archive,data,train_indices,hours,final_dir,lineage)
    for role in ['validation','independent_test']:
        destination=final_dir/role
        destination.mkdir(exist_ok=True)
        if role=='independent_test':
            manifest=dict(selection_sha256=file_sha256(selection_path),checkpoints={
                identifier(config,seed):file_sha256(final_dir/f'{identifier(config,seed)}.pt')
                for config in configs for seed in [42,17,101]})
            manifest_path=OUT/'independent_test_frozen_manifest.json'
            if manifest_path.exists():
                assert json.loads(manifest_path.read_text(encoding='utf-8'))==manifest
            else:
                manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        held,truth=final_truth(hours,years,role)
        results=[]
        for config in configs:
            for seed in [42,17,101]:
                model=train(config,seed,archive,data,train_indices,hours,final_dir,lineage)
                results.append(score(config,seed,model,archive,data,val,hours,held,truth,destination))
        pd.DataFrame(results).to_csv(destination/'comparison.csv',index=False)
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',selected=selected,
        independent_test_evaluated_after_selection=True,selection_sha256=file_sha256(selection_path),
        scope='best within registered candidate set; point prediction'),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
