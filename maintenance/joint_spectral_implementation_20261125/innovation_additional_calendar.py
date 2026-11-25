"""Frozen-head evaluation on a second 2020 calendar; no fitting or selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

from precision_analysis import ENGINE,statistics,bootstrap_macro,geography
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.p14_fusion import FusionData,P14Fusion
from tools.terrain_diffusion.innovation_experiment import feature_batch,CASES


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--day',type=int,default=15)
    args=p.parse_args();protocol=json.loads((args.experiment/'protocol.json').read_text(encoding='utf-8'))
    completed=json.loads((args.experiment/'status.json').read_text(encoding='utf-8'))
    if completed['completed']!=6 or protocol['config']['steps']!=3000:raise ValueError('requires frozen final two-seed fit')
    selection=json.loads((args.experiment/'frozen_selection.json').read_text(encoding='utf-8'))
    if selection['training_protocol_sha256']!=sha(args.experiment/'protocol.json') or selection['training_summary_sha256']!=sha(args.experiment/'summary.json'):raise ValueError('selection lock source changed')
    for name,expected in protocol['code_sha256'].items():
        if sha(ENGINE/'tools/terrain_diffusion'/name)!=expected:raise ValueError(f'frozen fit implementation changed: {name}')
    torch.set_num_threads(3);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.32)
    data_path=Path(protocol['config']['data']);source=Path(protocol['config']['source'])
    if sha(data_path/'contract.json')!=protocol['data_sha256']:raise ValueError('data contract changed')
    args.out.mkdir(parents=True,exist_ok=False);out=args.out
    weights={f'{name}_s{s}':sha(args.experiment/f'{name}_s{s}/coefficients.pt') for s in (42,17) for name in CASES}
    if weights!=selection['coefficient_sha256']:raise ValueError('coefficients changed after primary selection')
    write_json(out/'protocol.json',dict(day=args.day,hours=72,months=12,seeds=[42,17],
        purpose='Supplementary development calendar, no coefficient fitting or model selection',
        training_protocol_sha256=sha(args.experiment/'protocol.json'),coefficient_sha256=weights,
        code_sha256=sha(Path(__file__)),data_sha256=protocol['data_sha256'],
        source_weights=protocol['source_weights'],confirmation_C_D_read=False))
    write_json(out/'primary_selection.json',dict(primary_candidate=selection['primary_candidate'],
        selection_lock_sha256=sha(args.experiment/'frozen_selection.json'),created_utc=selection['created_utc']))
    data=FusionData(data_path);e=data.events;rows=[];draws={};regions={};indices=np.random.default_rng(252511).integers(0,12,(1000,12))
    nearest=[]
    for c in e.layouts:
        distance=np.linalg.norm(e.geo['dev_xy'][:,None]-e.geo['xy'][c][None],axis=-1);monthly=[]
        for month in range(1,13):
            epoch=int(np.datetime64(f'2020-{month:02d}-{args.day:02d}','h').astype('int64'))
            start=int(np.searchsorted(e.hours,epoch));valid=np.isfinite(e.values[start:start+72,c,0])
            monthly.append(np.where(valid[:,None,:],distance[None],np.inf).min(-1))
        nearest.append(monthly)
    nearest=np.asarray(nearest)
    for seed in (42,17):
        if sha(source/f'gru_s{seed}/last.pt')!=protocol['source_weights'][str(seed)]:raise ValueError('source changed')
        model=P14Fusion('gru').cuda().eval();model.load_state_dict(torch.load(source/f'gru_s{seed}/last.pt',weights_only=True))
        coefficients={n:torch.load(args.experiment/f'{n}_s{seed}/coefficients.pt',weights_only=True).cuda() for n in CASES}
        allpred={n:[] for n in ['reference',*CASES]};truth=[]
        for li,c in enumerate(e.layouts):
            monthly={n:[] for n in allpred}
            for month in range(1,13):
                epoch=int(np.datetime64(f'2020-{month:02d}-{args.day:02d}','h').astype('int64'))
                start=int(np.searchsorted(e.hours,epoch));parts={n:[] for n in allpred}
                for a in range(start,start+72,12):
                    x,pred,b,w=feature_batch(data,model,np.arange(a,a+12),c,True)
                    reference=pred*data.std[0]+data.mean[0];parts['reference'].append(reference.cpu().numpy())
                    for name,d in CASES.items():
                        correction=(x[...,:d].double()@coefficients[name]*10).clamp(-5,5)
                        parts[name].append((reference+correction).float().cpu().numpy())
                for name in monthly:monthly[name].append(np.concatenate(parts[name]))
                if li==0:
                    k=int(np.searchsorted(e.dev_hours,epoch));truth.append(np.array(e.dev_values[k:k+72,:,0]))
            for name in allpred:allpred[name].append(np.stack(monthly[name]))
            print(json.dumps(dict(seed=seed,layout=li,day=args.day)),flush=True)
        truth=np.stack(truth);predictions={n:np.stack(v) for n,v in allpred.items()}
        reference=statistics(predictions['reference'],truth)
        refdraw=bootstrap_macro(predictions['reference'].astype('float64')-truth[None],indices)
        for name,prediction in predictions.items():
            folder=out/f'{name}_s{seed}';folder.mkdir();np.savez_compressed(folder/'evaluation.npz',prediction=prediction,truth=truth)
            metrics=statistics(prediction,truth)
            draws[name,seed]=bootstrap_macro(prediction.astype('float64')-truth[None],indices)-refdraw
            row=dict(case=name,seed=seed,**metrics,
                delta_macro_rmse_mm=metrics['macro_rmse_mm']-reference['macro_rmse_mm'],
                delta_increment_macro_rmse_mm=metrics['increment_macro_rmse_mm']-reference['increment_macro_rmse_mm'])
            rows.append(row);write_json(folder/'metrics.json',row)
            regions[f'{name}_s{seed}']=geography(prediction,truth,nearest,e.geo)
        del model;torch.cuda.empty_cache()
    means=[]
    for name in ['reference',*CASES]:
        subset=[r for r in rows if r['case']==name];interval=np.quantile((draws[name,42]+draws[name,17])/2,[.025,.975]).tolist()
        means.append(dict(case=name,macro_rmse_mm=float(np.mean([r['macro_rmse_mm'] for r in subset])),
            micro_rmse_mm=float(np.mean([r['micro_rmse_mm'] for r in subset])),
            mae_mm=float(np.mean([r['mae_mm'] for r in subset])),
            increment_macro_rmse_mm=float(np.mean([r['increment_macro_rmse_mm'] for r in subset])),
            delta_macro_rmse_mm=float(np.mean([r['delta_macro_rmse_mm'] for r in subset])),paired95_mm=interval,
            improves_both_seeds=all(r['delta_macro_rmse_mm']<0 for r in subset),
            max_seed_increment_change_mm=max(r['delta_increment_macro_rmse_mm'] for r in subset)))
    write_json(out/'results.json',dict(records=rows,means=means,coefficient_sha256=weights,no_fitting=True))
    write_json(out/'geographic_and_pair_metrics.json',regions)
    lines=[f'# 冻结模型补充日历：2020每月{args.day}–{args.day+2}日','',
        '使用原GRU+p14及同一I0/I1/I2系数，所有系数来自2016–2019。此处只评价，不拟合、不按新增日历选择检查点。原8–10日主评价单独报告，主候选已在读取补充日期前冻结。','',
        '| 模型 | 双种子宏RMSE/mm | 微RMSE/mm | MAE/mm | 小时增量/mm | 宏RMSE差/mm | 月份配对95%区间/mm |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in means:lines.append(f"| {r['case']} | {r['macro_rmse_mm']:.6f} | {r['micro_rmse_mm']:.6f} | {r['mae_mm']:.6f} | {r['increment_macro_rmse_mm']:.6f} | {r['delta_macro_rmse_mm']:.6f} | {r['paired95_mm']} |")
    lines+=['','该补充仍属于2020年开发评价，137登记站、3布局、12个72小时连续块；并非完整年度或空间确认集。所有指标在float64中复算。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    write_json(out/'status.json',dict(status='COMPLETE_FROZEN_ADDITIONAL_CALENDAR',models=8))
    print(json.dumps(means),flush=True)


if __name__=='__main__':main()
