"""Read-only reuse of original Model4 algorithms, with a filtered training copy."""
import argparse
import json
import pickle
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch

from .data import SOURCE,HERE,CACHE,write_json,sha


def filter_h5(source,dest,stations):
    with h5py.File(source,'r') as f,h5py.File(dest,'x') as out:
        mask=np.isin(f['station_phys'].asstr()[:],stations)
        for key in f:
            out.create_dataset(key,data=f[key][:][mask],dtype=f[key].dtype,compression='gzip')
        for key,value in f.attrs.items():out.attrs[key]=value


def modules():
    sys.path.insert(0,str(SOURCE/'src'))
    from model4_fft_pipeline import baseline_workflow,config,data,lstm,xgb_stage
    return baseline_workflow,config,data,lstm,xgb_stage


def prepare():
    baseline,_,_,_,_=modules();out=CACHE/'original';out.mkdir(exist_ok=True)
    train=out/'train_273.h5'
    if not train.exists():
        stations=pd.read_csv(HERE/'stations.csv');filter_h5(SOURCE/'data/euref_train_2020_2023.h5',train,stations.loc[stations.role=='train','station_phys'])
    if not (out/'baseline.pkl').exists():
        print(json.dumps(dict(stage='original_baseline_fit',status='starting')),flush=True)
        result=baseline.fit_baseline_from_h5(train,output_dir=out/'baseline',spatial_method='scha')
        with (out/'baseline.pkl').open('wb') as f:pickle.dump(result.spatial_model,f)
        write_json(HERE/'original_baseline_receipt.json',dict(train_h5_sha256=sha(train),model_sha256=sha(out/'baseline.pkl'),original_baseline_code_sha256=sha(SOURCE/'src/model4_fft_pipeline/baseline_workflow.py'),stations=273,years=[2020,2021,2022,2023],spatial_method='original SCHA default with spatial CV'))
    print(json.dumps(dict(stage='original_baseline_fit',status='complete')),flush=True)


class Progress:
    def __init__(self,out):self.out=out
    def report(self,value,step):
        row=dict(epoch=step+1,val_loss=float(value))
        write_json(self.out/'progress.json',row);print(json.dumps(dict(stage=self.out.name,**row)),flush=True)
    def should_prune(self):return False


def run(seed=42):
    _,config,data,lstm,xgb=modules();torch.set_num_threads(4)
    local=CACHE/'original';out=HERE/'runs'/f'original_s{seed}';out.mkdir(parents=True,exist_ok=False)
    modeldir=local/f'models_s{seed}';modeldir.mkdir(exist_ok=True)
    cfg=config.PipelineConfig.normal(root=SOURCE,train_h5=local/'train_273.h5',test_h5=SOURCE/'data/euref_test_2024.h5',models_dir=modeldir,results_dir=modeldir/'results',figures_dir=modeldir/'figures',random_state=seed)
    config.ensure_dirs(cfg)
    with (local/'baseline.pkl').open('rb') as f:provider=pickle.load(f)
    lock=dict(seed=seed,train_station_csv_sha256=sha(HERE/'stations.csv'),train_h5_sha256=sha(cfg.train_h5),test_h5_sha256=sha(cfg.test_h5),config={k:str(v) if hasattr(v,'as_posix') else v for k,v in vars(cfg).items()},sources={name:sha(SOURCE/'src/model4_fft_pipeline'/name) for name in ['lstm.py','xgb_stage.py','contracts.py','data.py','scha.py','harmonic.py']},new_model_features_mixed=False)
    write_json(out/'lock.json',lock);start=time.monotonic()
    full=data.prepare_dataframe(cfg.train_h5,baseline_provider=provider,config=cfg)
    train,val=data.split_lstm_time(full)
    bundle=lstm.train_lstm_model(train,val,cfg,optuna_trial=Progress(out))
    lstm.save_lstm_checkpoint(bundle,cfg.lstm_model_path)
    write_json(out/'learning_curve.json',bundle['history'])
    valpred=lstm.predict_lstm(bundle,val,cfg)
    model,diagnostics,_=xgb.train_xgb_residual(valpred,cfg)
    model.save_model(cfg.xgb_model_path)
    write_json(out/'xgb_diagnostics.json',diagnostics)
    # 2024 read after all fitting and early stopping have finished.
    test=data.prepare_dataframe(cfg.test_h5,baseline_provider=provider,config=cfg)
    testpred=xgb.predict_xgb(model,lstm.predict_lstm(bundle,test,cfg))
    meta=pd.read_csv(HERE/'stations.csv');station=pd.Index(meta.station_phys).get_indexer(testpred.station_phys)
    hours=pd.to_datetime(testpred.datetime).to_numpy().astype('datetime64[h]').astype('i8')
    np.savez_compressed(local/f'predictions_s{seed}.npz',hours=hours,station=station,pred=testpred.zwd_pred.to_numpy()*1000,target=testpred.zwd.to_numpy()*1000,lstm_pred=(testpred.baseline_pred+testpred.lstm_pred).to_numpy()*1000)
    write_json(out/'complete.json',dict(seconds=time.monotonic()-start,best_epoch=bundle['best_epoch'],train_sequences=bundle['train_sequences'],validation_sequences=bundle['validation_sequences'],finite_2024_predictions=int(testpred.zwd_pred.notna().sum()),lstm_sha256=sha(cfg.lstm_model_path),xgb_sha256=sha(cfg.xgb_model_path),prediction_sha256=sha(local/f'predictions_s{seed}.npz')))
    print(json.dumps(dict(stage=out.name,status='complete')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if args.prepare_only:prepare()
    else:run(args.seed)
