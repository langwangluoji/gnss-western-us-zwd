"""Explicit commands; only `run` executes the complete registered pipeline."""
import argparse
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager, ExitStack
import torch
from .data import audit, cache_mean, ExperimentData, DEFAULT_ROOT
from .training import write_json, search
from .workflow import smoke, promote, replicate, evaluate_locked, report, ARMS


@contextmanager
def stage_lock(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a+b') as guard:
        guard.seek(0); guard.write(b'0'); guard.flush(); guard.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','cache-mean','smoke','search','promote','replicate','evaluate','report','early-mean','run'])
    parser.add_argument('--stage',choices=['stage1','stage2'],default='stage1')
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--root',type=Path,help='Output directory for this stage')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=args.root or DEFAULT_ROOT/args.stage
    root.mkdir(parents=True,exist_ok=True)
    # OS-held file lock releases automatically after process termination; unlike
    # PID-only sentinels it cannot mistake a recycled PID for the old worker.
    with ExitStack() as locks:
        locks.enter_context(stage_lock(root))
        if args.command=='report':
            print(report(root)); return
        from .data import sha256
        code_identity={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        identity_path=root/'code_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8'))!=code_identity:
            raise ValueError('Experiment code changed; inspect and register a new output root before continuing')
        write_json(identity_path,code_identity)
        import optuna
        write_json(root/'runtime.json',dict(python=sys.executable,torch=torch.__version__,optuna=optuna.__version__,
             cuda=torch.version.cuda,device=args.device,threads=2))
        if args.command=='audit':
            print(json.dumps(audit(args.stage,root),ensure_ascii=False)); return
        if args.command=='early-mean':
            from .early_mean import train_early_mean
            print(train_early_mean(root,args.device)); return
        if args.command=='cache-mean':
            print(cache_mean(args.stage,root,args.device)); return
        if args.command=='run':
            stages=[(args.stage,root)]
            run_stage(args.stage,root,args.device)
            if args.stage=='stage1':
                replication=json.loads((root/'replication.json').read_text(encoding='utf-8'))
                if replication.get('gate',{}).get('passed'):
                    second=root.parent/'stage2'
                    locks.enter_context(stage_lock(second))
                    if (second/'code_identity.json').exists() and json.loads((second/'code_identity.json').read_text(encoding='utf-8'))!=code_identity:
                        raise ValueError('Stage2 code identity changed')
                    write_json(second/'code_identity.json',code_identity)
                    from .early_mean import train_early_mean
                    train_early_mean(second,args.device)
                    run_stage('stage2',second,args.device,root)
                    stages.append(('stage2',second))
            # Read retrospective observations only after every reached stage has
            # locked its development choices. No future search consumes results.
            for stage,folder in stages:
                data=ExperimentData(stage,folder,args.device)
                data.preload_inputs()
                try:
                    write_json(folder/'status.json',dict(state='retrospective',stage=stage,pid=os.getpid()))
                    evaluate_locked(data,folder)
                    write_json(folder/'status.json',dict(state='complete',stage=stage,pid=os.getpid()))
                    report(folder)
                finally: data.close()
            return
        data=ExperimentData(args.stage,root,args.device)
        data.preload_inputs()
        if args.command=='smoke': print(smoke(data,root))
        elif args.command=='search':
            if not (root/'smoke.json').exists(): raise RuntimeError('Run smoke before search')
            if not args.arm: parser.error('search requires --arm')
            first=None
            if args.stage=='stage2':
                first=json.loads((root.parent/'stage1/selection.json').read_text(encoding='utf-8'))['candidates'][args.arm][0]['config']
            search(data,root,args.arm,args.stage,first)
        elif args.command=='promote': promote(data,root,args.stage)
        elif args.command=='replicate': replicate(data,root)
        elif args.command=='evaluate': evaluate_locked(data,root)
        print(report(root))


def run_stage(stage,root,device,previous=None):
    root=Path(root)
    data=None
    try:
        write_json(root/'status.json',dict(state='audit',stage=stage,pid=os.getpid()))
        audit(stage,root)
        write_json(root/'status.json',dict(state='cache_mean',stage=stage,pid=os.getpid()))
        cache_mean(stage,root,device)
        data=ExperimentData(stage,root,device)
        cached_bytes=data.preload_inputs()
        write_json(root/'input_memory.json',dict(bytes=cached_bytes,policy='read-only inputs; labels remain lazy'))
        write_json(root/'status.json',dict(state='smoke',stage=stage,pid=os.getpid()))
        smoke(data,root)
        for arm in ARMS:
            write_json(root/'status.json',dict(state='search',stage=stage,arm=arm,pid=os.getpid()))
            first=None if previous is None else json.loads((Path(previous)/'selection.json').read_text(encoding='utf-8'))['candidates'][arm][0]['config']
            search(data,root,arm,stage,first)
        write_json(root/'status.json',dict(state='promote',stage=stage,pid=os.getpid()))
        promote(data,root,stage)
        write_json(root/'status.json',dict(state='replicate',stage=stage,pid=os.getpid()))
        replicate(data,root)
        write_json(root/'status.json',dict(state='development_locked',stage=stage,pid=os.getpid()))
    except BaseException as exc:
        write_json(root/'status.json',dict(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            stage=stage,error=type(exc).__name__+': '+str(exc),pid=os.getpid()))
        raise
    finally:
        if data is not None: data.close()
        if root.exists(): report(root)


if __name__=='__main__': main()
