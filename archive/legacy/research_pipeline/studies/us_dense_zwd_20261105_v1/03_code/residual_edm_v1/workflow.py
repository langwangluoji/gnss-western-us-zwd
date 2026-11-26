"""Registered promotion gates and explicit, recoverable experiment stages."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from .training import write_json, train_updates, load_checkpoint, fingerprint

ARMS = ('det','gaussian','edm')
SAMPLE_SEED = 20261113


def validate_promotion(study,stage):
    import optuna
    budget=12 if stage=='stage1' else 6
    terminal=(optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.FAIL,optuna.trial.TrialState.PRUNED)
    if len(study.trials)!=budget or any(t.state not in terminal for t in study.trials):
        raise ValueError('Complete the registered search budget before promotion')


def validate_evaluation(selection,replication):
    if replication.get('status')=='not_promoted' and not selection['gate']['passed']:
        return
    if replication.get('status')!='complete':
        raise ValueError('Retrospective evaluation requires terminal replication or a failed selection gate')
    for arm in ARMS:
        if {r['seed'] for r in replication.get('records',{}).get(arm,[])}!={42,17,101}:
            raise ValueError('Retrospective evaluation requires all registered replication seeds')


def gate(mean_rmse, det_scores, edm_scores):
    d, e = np.asarray(det_scores), np.asarray(edm_scores)
    if d.shape != e.shape or not len(e) or not np.isfinite([mean_rmse,*d,*e]).all():
        raise ValueError('Gate requires finite paired seed scores')
    anchor = float(mean_rmse-e.mean())
    deterministic = float((d-e).mean())
    worst = float((e-d).max())
    return dict(passed=bool(anchor>=.05-1e-12 and deterministic>=.02-1e-12 and worst<=.05+1e-12),
                mean_gain_mm=anchor, deterministic_gain_mm=deterministic,
                worst_seed_regression_mm=worst, thresholds=dict(mean=.05,deterministic=.02,worst=.05),
                meaning='pre-registered engineering gate, not a significance test')


def mc_stable(mean_rmse, det_rmse, previous, current):
    return bool(abs(current-previous)<=.005 and
                gate(mean_rmse,[det_rmse],[previous])['passed']==gate(mean_rmse,[det_rmse],[current])['passed'])


def model_from_checkpoint(path, device):
    from .model import ResidualModel
    state = load_checkpoint(path)
    model = ResidualModel(state['contract']['arm']).to(device)
    model.load_state_dict(state['ema'])
    return model.eval()


def measured(model, data, indices, destination, members=32, role='validation', years=(2019,), probability=False):
    from .evaluation import evaluate
    destination=Path(destination)
    registration=dict(data=fingerprint(data.manifest), indices=fingerprint(np.asarray(indices).tolist()),
        members=members,role=role,years=list(years),probability=probability,seed=SAMPLE_SEED)
    if model is not None:
        import hashlib
        digest=hashlib.sha256()
        for key,value in model.state_dict().items():
            digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        registration['model']=digest.hexdigest()
    record=destination/'result.json'
    if record.exists():
        saved=json.loads(record.read_text(encoding='utf-8'))
        if saved['registration']!=registration: raise ValueError('Evaluation cache contract changed')
        return saved['metrics']
    metrics=evaluate(model,data,indices,members=members,seed=SAMPLE_SEED,output=destination,
                     role=role,years=years,probability=probability)
    write_json(record,dict(registration=registration,metrics=metrics))
    return metrics


def smoke(data, root):
    from .model import ResidualModel
    from .training import rng_state, restore_rng
    root=Path(root)
    if (root/'smoke.json').exists():
        saved=json.loads((root/'smoke.json').read_text(encoding='utf-8'))
        if saved['data']!=fingerprint(data.manifest): raise ValueError('Smoke data changed')
        return saved
    class Small:
        train_indices=data.train_indices[:16]
        manifest=dict(data.manifest,smoke_indices=train_indices.tolist())
        def batch(self,ids,targets=True): return data.batch(ids,targets)
    result={}
    batch=data.batch(Small.train_indices)
    for arm in ARMS:
        torch.manual_seed(42)
        model=ResidualModel(arm).to(data.device)
        fixed=rng_state()
        model.eval()
        with torch.no_grad(): initial=float(model.loss(batch))
        config=dict(lr=1e-4,weight_decay=.01,ema=.999)
        train_updates(model,Small(),root/'smoke'/arm,config,42,200,eval_every=1000)
        model.eval(); restore_rng(fixed)
        with torch.no_grad(): final=float(model.loss(batch))
        if not np.isfinite(final) or final>=initial:
            raise RuntimeError(f'Smoke did not learn: {arm} {initial} -> {final}')
        result[arm]=dict(initial_loss=initial,final_loss=final)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    saved=dict(data=fingerprint(data.manifest),passed=True,arms=result)
    write_json(root/'smoke.json',saved)
    return saved


def promote(data,root,stage):
    import optuna
    from .evaluation import evaluate
    from .model import ResidualModel
    root=Path(root)
    chosen={}
    for arm in ARMS:
        study=optuna.load_study(study_name=f'{stage}_{arm}',storage=f"sqlite:///{(root/'studies.sqlite3').resolve().as_posix()}")
        validate_promotion(study,stage)
        if any(t.state in (optuna.trial.TrialState.RUNNING,optuna.trial.TrialState.WAITING) for t in study.trials):
            raise RuntimeError('Search must finish before promotion')
        candidates=sorted([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE],key=lambda t:t.value)
        candidates=candidates[:3 if stage=='stage1' else 2]
        if not candidates: raise RuntimeError(f'No completed {arm} trials; diagnose study')
        records=[]
        for trial in candidates:
            directory=root/arm/f'trial_{trial.number:03d}'
            torch.manual_seed(42)
            model=ResidualModel(arm).to(data.device)
            train_updates(model,data,directory,trial.params,42,50000,eval_every=5000,
                          evaluator=lambda m:evaluate(m,data,data.screen_indices,members=8,seed=SAMPLE_SEED))
            model=model_from_checkpoint(directory/'best.pt',data.device)
            metrics=measured(model,data,data.dev_indices,directory/'full_development_K32',32)
            records.append(dict(trial=trial.number,checkpoint=str(directory/'best.pt'),config=trial.params,
                                score=metrics['station_macro_rmse_mm'],members=32))
            del model
        chosen[arm]=sorted(records,key=lambda r:r['score'])
    base=measured(None,data,data.dev_indices,root/'mean_development',1)['station_macro_rmse_mm']
    # Refine every shortlisted EDM candidate, preserving rankings and noise prefixes.
    stable=False
    previous=chosen['edm']
    for members in (64,128):
        current=[]
        for entry in previous:
            model=model_from_checkpoint(entry['checkpoint'],data.device)
            destination=Path(entry['checkpoint']).parent/f'full_development_K{members}'
            score=measured(model,data,data.dev_indices,destination,members)['station_macro_rmse_mm']
            current.append(dict(entry,score=score,members=members))
            del model
        current.sort(key=lambda r:r['score'])
        prior={r['trial']:r for r in previous}
        stable=(current[0]['trial']==previous[0]['trial'] and all(
            mc_stable(base,chosen['det'][0]['score'],prior[r['trial']]['score'],r['score']) for r in current))
        previous=current
        if stable: break
    chosen['edm']=previous
    result=dict(data=fingerprint(data.manifest),mean_score=base,candidates=chosen,mc_stable=stable,
                gate=gate(base,[chosen['det'][0]['score']],[chosen['edm'][0]['score']]))
    result['gate']['passed'] = bool(result['gate']['passed'] and stable)
    write_json(root/'selection.json',result)
    return result


def replicate(data,root):
    from .model import ResidualModel
    from .evaluation import evaluate
    root=Path(root)
    selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    if selection['data']!=fingerprint(data.manifest): raise ValueError('Selection data changed')
    if not selection['gate']['passed']:
        result=dict(status='not_promoted',gate=selection['gate'])
        write_json(root/'replication.json',result); return result
    records={arm:[dict(seed=42,**selection['candidates'][arm][0])] for arm in ARMS}
    stable=True
    for seed in (17,101):
        for arm in ARMS:
            config=selection['candidates'][arm][0]['config']
            directory=root/'replication'/arm/f'seed{seed}'
            torch.manual_seed(seed)
            model=ResidualModel(arm).to(data.device)
            train_updates(model,data,directory,config,seed,50000,eval_every=5000,
                          evaluator=lambda m:evaluate(m,data,data.screen_indices,members=8,seed=SAMPLE_SEED))
            model=model_from_checkpoint(directory/'best.pt',data.device)
            score=measured(model,data,data.dev_indices,directory/'full_development_K32',32)['station_macro_rmse_mm']
            members=32
            if arm=='edm':
                accepted=False
                for members in (64,128):
                    updated=measured(model,data,data.dev_indices,directory/f'full_development_K{members}',members)['station_macro_rmse_mm']
                    accepted=mc_stable(selection['mean_score'],records['det'][-1]['score'],score,updated)
                    score=updated
                    if accepted: break
                stable &= accepted
            records[arm].append(dict(seed=seed,score=score,config=config,checkpoint=str(directory/'best.pt'),members=members))
            del model
    decision=gate(selection['mean_score'],[r['score'] for r in records['det']],[r['score'] for r in records['edm']])
    decision['passed']=bool(decision['passed'] and stable)
    result=dict(status='complete',data=fingerprint(data.manifest),records=records,mc_stable=stable,gate=decision)
    write_json(root/'replication.json',result)
    return result


def evaluate_locked(data,root):
    from .data import sha256
    root=Path(root)
    selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    replication=json.loads((root/'replication.json').read_text(encoding='utf-8')) if (root/'replication.json').exists() else {}
    validate_evaluation(selection,replication)
    records=replication.get('records',{arm:[dict(seed=42,**selection['candidates'][arm][0])] for arm in ARMS})
    entries={f'{arm}_seed{r["seed"]}':dict(r,arm=arm,sha256=sha256(r['checkpoint'])) for arm,rows in records.items() for r in rows}
    lock=dict(data=fingerprint(data.manifest),entries=entries,selection=fingerprint(selection),
              boundary='2020-2021 retrospective; no further tuning permitted')
    lock_path=root/'evaluation_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text(encoding='utf-8'))!=lock:
        raise ValueError('Evaluation lock changed')
    write_json(lock_path,lock)
    years=np.asarray(data.years)
    indices=np.asarray(data.all_indices)
    retro=indices[np.isin(years[indices],[2020,2021])]
    outputs={}
    for role in ('validation','independent_test'):
        base=measured(None,data,retro,root/'retrospective'/role/'mean',1,role,(2020,2021),True)
        outputs[f'{role}_mean']=base
        for name,entry in entries.items():
            if sha256(entry['checkpoint'])!=entry['sha256']: raise ValueError('Locked weights changed')
            model=model_from_checkpoint(entry['checkpoint'],data.device)
            outputs[f'{role}_{name}']=measured(model,data,retro,root/'retrospective'/role/name,
                max(32,entry['members']),role,(2020,2021),True)
            del model
    write_json(root/'retrospective_results.json',outputs)
    paired_results={}
    import h5py
    from .evaluation import paired_block_bootstrap
    for role in ('validation','independent_test'):
        for edm_name,entry in entries.items():
            if entry['arm']!='edm': continue
            for comparator in ('mean',f'det_seed{entry["seed"]}'):
                first=root/'retrospective'/role/comparator/'evaluation.h5'
                second=root/'retrospective'/role/edm_name/'evaluation.h5'
                with h5py.File(first,'r') as a,h5py.File(second,'r') as b:
                    for key in ('hour_id','valid','truth_mm'):
                        if not np.array_equal(a[key][:],b[key][:],equal_nan=True):
                            raise ValueError('Paired evaluation axes/support disagree')
                    paired_results[f'{role}_{comparator}_vs_{edm_name}']=paired_block_bootstrap(
                        a['prediction_mm'][:]-a['truth_mm'][:],b['prediction_mm'][:]-b['truth_mm'][:],
                        a['valid'][:],a['hour_id'][:])
    write_json(root/'paired_bootstrap.json',paired_results)
    return outputs


def report(root):
    root=Path(root)
    lines=['# Residual EDM experiment status','',
           'Development: 2019. Retrospective evaluation: 2020–2021. No blind-test claim.','']
    for filename in ('smoke.json','selection.json','replication.json','retrospective_results.json'):
        path=root/filename
        if path.exists():
            value=json.loads(path.read_text(encoding='utf-8'))
            if filename=='selection.json':
                lines += ['## Selected configurations',f"Frozen mean macro RMSE: {value['mean_score']:.6f} mm",'']
                for arm,rows in value['candidates'].items():
                    lines.append(f"- {arm}: {rows[0]['score']:.6f} mm; K={rows[0]['members']}; trial {rows[0]['trial']}")
                lines += ['',f"MC stable: {value['mc_stable']}; engineering gate: {value['gate']}",'']
            elif filename=='retrospective_results.json':
                lines += ['## Retrospective results','']
                for name,metrics in value.items():
                    lines.append(f"- {name}: {metrics['station_macro_rmse_mm']:.6f} mm; records={metrics['station_records']}")
            else:
                lines += [f"{filename}: {value.get('status',value.get('passed'))}; gate={value.get('gate','not applicable')}",'']
        else: lines.append(f'{filename}: pending')
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return root/'report.md'
