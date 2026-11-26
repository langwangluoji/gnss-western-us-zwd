"""Original CNN-LSTM-XGBoost + its original three VMF3 features."""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch

from .features import HERE,CACHE,PARENT,OLD_CACHE,SOURCE,sha,write_json
from maintenance.europe_transfer_20261126.legacy import modules,Progress


def run(seed):
    _,config,data,lstm,xgb=modules()
    from model4_fft_pipeline.contracts import enable_vmf3_background,LSTM_FEATURE_COLS
    torch.set_num_threads(4);enable_vmf3_background()
    out=HERE/'runs'/f'cnn_vmf3_s{seed}';out.mkdir(parents=True,exist_ok=False)
    modeldir=CACHE/f'cnn_models_s{seed}';modeldir.mkdir(exist_ok=True)
    cfg=config.PipelineConfig.normal(root=SOURCE,train_h5=OLD_CACHE/'original/train_273.h5',test_h5=SOURCE/'data/euref_test_2024.h5',models_dir=modeldir,results_dir=modeldir/'results',figures_dir=modeldir/'figures',random_state=seed,use_vmf3_background=True)
    config.ensure_dirs(cfg)
    baseline=OLD_CACHE/'original/baseline.pkl';baseline_receipt=json.loads((PARENT/'original_baseline_receipt.json').read_text(encoding='utf-8'));assert sha(baseline)==baseline_receipt['model_sha256']
    with baseline.open('rb') as f:provider=pickle.load(f)
    write_json(out/'lock.json',dict(seed=seed,train_h5_sha256=sha(cfg.train_h5),test_h5_sha256=sha(cfg.test_h5),baseline_sha256=sha(baseline),features=LSTM_FEATURE_COLS,vmf3_enabled=True,training_station_csv_sha256=sha(PARENT/'stations.csv'),config={k:str(v) if hasattr(v,'as_posix') else v for k,v in vars(cfg).items()},original_code={name:sha(SOURCE/'src/model4_fft_pipeline'/name) for name in ['data.py','lstm.py','xgb_stage.py','vmf3_background.py','contracts.py']}))
    start=time.monotonic();meta={}
    full=data.prepare_dataframe(cfg.train_h5,baseline_provider=provider,config=cfg,vmf3_meta_out=meta.setdefault('train',{}));train,val=data.split_lstm_time(full)
    bundle=lstm.train_lstm_model(train,val,cfg,optuna_trial=Progress(out));lstm.save_lstm_checkpoint(bundle,cfg.lstm_model_path)
    write_json(out/'learning_curve.json',bundle['history']);valpred=lstm.predict_lstm(bundle,val,cfg)
    model,diagnostics,_=xgb.train_xgb_residual(valpred,cfg);model.save_model(cfg.xgb_model_path);write_json(out/'xgb_diagnostics.json',diagnostics)
    test=data.prepare_dataframe(cfg.test_h5,baseline_provider=provider,config=cfg,vmf3_meta_out=meta.setdefault('test',{}));testpred=xgb.predict_xgb(model,lstm.predict_lstm(bundle,test,cfg))
    stations=pd.read_csv(PARENT/'stations.csv');index=pd.Index(stations.station_phys).get_indexer(testpred.station_phys);hours=pd.to_datetime(testpred.datetime).to_numpy().astype('datetime64[h]').astype('i8')
    path=CACHE/f'cnn_predictions_s{seed}.npz';np.savez_compressed(path,hours=hours,station=index,pred=testpred.zwd_pred.to_numpy()*1000,target=testpred.zwd.to_numpy()*1000)
    write_json(out/'vmf3_meta.json',meta);write_json(out/'complete.json',dict(seconds=time.monotonic()-start,best_epoch=bundle['best_epoch'],train_sequences=bundle['train_sequences'],validation_sequences=bundle['validation_sequences'],finite_2024_predictions=int(testpred.zwd_pred.notna().sum()),predictions_sha256=sha(path),lstm_sha256=sha(cfg.lstm_model_path),xgb_sha256=sha(cfg.xgb_model_path)))
    print(json.dumps(dict(stage=out.name,status='complete')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);run(p.parse_args().seed)
