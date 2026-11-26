"""Serialized, reconcilable Slurm submission with a cumulative four-card-hour cap."""
import argparse
import getpass
import os
from pathlib import Path
import subprocess
import uuid

from gnss_field.artifacts import read_json,write_json,immutable_json,atomic_bytes,sha256

ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED'}


def remaining_allocation(attempts,records,cap=14400):
    consumed=0
    for row in attempts:
        state=records.get(row['job_id'])
        if state is None or state.get('elapsed') is None:
            raise ValueError('Accounting unavailable for a prior allocation; refusing to grant fresh budget')
        consumed+=int(state['elapsed'])
    return max(0,cap-consumed)


def scheduler_records():
    user=getpass.getuser(); records={}
    result=subprocess.run(['sacct','-X','-u',user,'-S','2026-11-21','-n','-P',
        '-o','JobIDRaw%30,State%32,ElapsedRaw,JobName%128'],capture_output=True,text=True,check=True,timeout=30)
    for line in result.stdout.splitlines():
        columns=line.split('|')
        if len(columns)<4 or not columns[0].strip().isdigit(): continue
        jid,state,elapsed,name=columns[:4]
        records[jid.strip()]=dict(state=state.strip().split()[0].rstrip('+'),
            elapsed=int(elapsed),name=name.strip(),partition='')
    result=subprocess.run(['squeue','-u',user,'-h','-o','%i|%T|%j|%P'],
        capture_output=True,text=True,check=True,timeout=30)
    for line in result.stdout.splitlines():
        jid,state,name,partition=line.split('|',3)
        old=records.setdefault(jid,dict(elapsed=None))
        old.update(state=state,name=name,partition=partition)
    return records


def submit(root,release,python_bin,*,resume=False):
    import fcntl
    from grid_pair_release import verify_grid_release
    root,release=Path(root),Path(release)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        immutable_json(root/'allocation.lock.json',dict(cap_seconds=14400,gpus=1,
            release_sha256=verify_grid_release(release)))
        history=read_json(root/'allocations.json') if (root/'allocations.json').exists() else []
        records=scheduler_records()
        intent_path=root/'submission_intent.json'
        if intent_path.exists():
            intent=read_json(intent_path)
            if intent['state'] in ('SUBMITTING','UNKNOWN'):
                found=[jid for jid,s in records.items() if s.get('name')==intent['token']]
                if len(found)!=1:
                    raise ValueError('Uncertain submission cannot yet be reconciled; no duplicate submitted')
                jid=found[0]
                if not any(row['job_id']==jid for row in history):
                    history.append(dict(job_id=jid,token=intent['token']))
                    write_json(root/'allocations.json',history)
                write_json(intent_path,dict(**{k:v for k,v in intent.items() if k!='state'},state='REGISTERED'))
                atomic_bytes(root/'job_id',(jid+'\n').encode())
        active=[row['job_id'] for row in history if records.get(row['job_id'],{}).get('state') in ACTIVE]
        if active:
            if len(active)!=1: raise ValueError('Multiple active own allocations; no new job')
            atomic_bytes(root/'job_id',(active[0]+'\n').encode())
            return dict(status='EXISTING_ACTIVE',job_id=active[0])
        if (root/'queue_status.json').exists() and read_json(root/'queue_status.json')['status']=='COMPLETE':
            return dict(status='COMPLETE',job_id=history[-1]['job_id'] if history else None)
        if history and not resume:
            raise ValueError('Prior job is terminal; use explicit --resume to continue its remaining budget')
        remaining=remaining_allocation(history,records)
        minutes=remaining//60
        if minutes<4:
            write_json(root/'queue_status.json',dict(status='TOTAL_BUDGET_EXHAUSTED',remaining_seconds=remaining))
            raise ValueError('Cumulative allocation cap exhausted')
        if any(s.get('partition','').upper()=='A800' and s['state'] in ACTIVE for s in records.values()):
            raise ValueError('Existing A800 work preserved; no additional GPU submitted')
        if resume and (root/'PAUSE').exists(): (root/'PAUSE').unlink()
        token='ZSY_gen3_'+uuid.uuid4().hex[:12]
        intent=dict(state='SUBMITTING',token=token,remaining_seconds=remaining,
            wall_minutes=minutes,attempt=len(history)+1)
        write_json(intent_path,intent)
        env=os.environ.copy()
        env.update(GRID_RELEASE=str(release),GRID_RUN=str(root),GRID_PYTHON=str(python_bin),
                   GRID_ALLOWED_SECONDS=str(minutes*60-180))
        command=['sbatch','--parsable','--job-name='+token,'--time='+str(minutes),
            '--output='+str(root/'logs/slurm_%j.out'),str(release/'slurm/grid_generation_a800.sh')]
        try:
            result=subprocess.run(command,env=env,capture_output=True,text=True,check=True,timeout=30)
        except subprocess.CalledProcessError as error:
            write_json(intent_path,dict(**{k:v for k,v in intent.items() if k!='state'},
                state='FAILED',error=error.stderr[-4000:]))
            raise
        except (subprocess.TimeoutExpired,OSError):
            write_json(intent_path,dict(**{k:v for k,v in intent.items() if k!='state'},state='UNKNOWN'))
            raise
        jid=result.stdout.strip().split(';')[0]
        if not jid.isdigit(): raise ValueError('Unrecognized sbatch receipt; intent kept for reconciliation')
        history.append(dict(job_id=jid,token=token))
        write_json(root/'allocations.json',history)
        atomic_bytes(root/'job_id',(jid+'\n').encode())
        write_json(intent_path,dict(**{k:v for k,v in intent.items() if k!='state'},state='REGISTERED',job_id=jid))
        return dict(status='SUBMITTED',job_id=jid,remaining_seconds=remaining,
                    max_parallel=1,wall_minutes=minutes)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True); p.add_argument('--python',required=True)
    p.add_argument('--resume',action='store_true'); a=p.parse_args()
    print(submit(a.root,a.release,a.python,resume=a.resume),flush=True)
