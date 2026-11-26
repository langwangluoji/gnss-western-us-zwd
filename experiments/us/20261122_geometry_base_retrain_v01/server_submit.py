"""Single-A800 submission, one-hour cumulative allocation budget, no automatic requeue."""
import argparse,json,os,subprocess,getpass,time,uuid
from pathlib import Path


def remaining_budget(elapsed):
    if any(int(v)<0 for v in elapsed):raise ValueError('Negative allocation accounting')
    return max(0,3600-sum(int(v) for v in elapsed))


def terminal_accounting(state):
    return state.split()[0].rstrip('+') in {'COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def command(args):return subprocess.run(args,capture_output=True,text=True,check=True,timeout=40).stdout.strip()


def submit(release,root,resume=False):
    import fcntl
    from gnss_field.artifacts import read_json,write_json,sha256,immutable_json
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    with (root/'.submit.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        immutable_json(root/'allocation.lock.json',dict(cap_seconds=3600,gpus=1,release=sha256(release/'code_manifest.json'),data=sha256(release/'server_data_manifest.json')))
        history=read_json(root/'allocations.json') if (root/'allocations.json').exists() else []
        queues=[line.split('|') for line in command(['squeue','-u',getpass.getuser(),'-h','-o','%i|%P|%T|%j']).splitlines() if line.strip()]
        active={row[0]:row for row in queues}
        accounting={}
        for line in command(['sacct','-X','-u',getpass.getuser(),'-S','now-2days','-n','-P','-o','JobIDRaw%30,ElapsedRaw,State%32,JobName%128']).splitlines():
            columns=line.split('|')
            if len(columns)>=4 and columns[0].strip().isdigit():accounting[columns[0].strip()]=dict(elapsed=int(columns[1]),state=columns[2].strip(),name=columns[3].strip())
        intent_path=root/'submission_intent.json'
        if intent_path.exists():
            intent=read_json(intent_path)
            if intent['state'] in ('SUBMITTING','UNKNOWN'):
                found=set(j for j,row in active.items() if row[3]==intent['token'])|set(j for j,row in accounting.items() if row['name']==intent['token'])
                if len(found)!=1:raise ValueError('Uncertain prior submission; refusing a duplicate allocation')
                jid=found.pop()
                if not any(h['job_id']==jid for h in history):history.append(dict(job_id=jid,token=intent['token']));write_json(root/'allocations.json',history)
                write_json(intent_path,{**intent,'state':'REGISTERED'})
        for row in history:
            if row['job_id'] in active:
                (root/'job_id').write_text(row['job_id']+'\n');return dict(state='EXISTING_ACTIVE',job_id=row['job_id'])
        status_path=root/'status.json'
        if status_path.exists() and read_json(status_path).get('stage')=='COMPLETE':return dict(state='COMPLETE',history=history)
        if history and not resume:raise ValueError('Prior job terminal; explicit --resume is required, with same remaining budget')
        elapsed=[]
        for row in history:
            if row['job_id'] not in accounting:raise ValueError('Prior allocation accounting missing; cannot grant new budget')
            if not terminal_accounting(accounting[row['job_id']]['state']):raise ValueError('Prior job accounting is not terminal; refusing overlap or fresh allowance')
            elapsed.append(accounting[row['job_id']]['elapsed'])
        remaining=remaining_budget(elapsed);minutes=remaining//60
        if minutes<4:raise ValueError('One-card-hour cumulative budget exhausted')
        template=(release/'results/20261122_geometry_base_retrain_v01/A800.sh').read_text()
        script=root/'A800.sh';script.write_text(template)
        token='ZSY_geo_'+uuid.uuid4().hex[:12];intent=dict(state='SUBMITTING',token=token,remaining_seconds=remaining);write_json(intent_path,intent)
        dependencies=[row[0] for row in queues if row[1].upper()=='A800']
        args=['sbatch','--parsable','--no-requeue','--export=ALL','--time='+str(minutes),'--job-name='+token,'--output='+str(root/'logs/slurm_%j.out'),'--error='+str(root/'logs/slurm_%j.err')]
        if dependencies:args+=['--dependency=afterany:'+':'.join(dependencies)]
        env=os.environ.copy();env.update(GNSS_RELEASE=str(release),GNSS_RETRAIN_ROOT=str(root),GNSS_ALLOWED_SECONDS=str(min(3420,minutes*60-180)))
        try:jid=subprocess.run(args+[str(script)],env=env,capture_output=True,text=True,check=True,timeout=40).stdout.strip().split(';')[0]
        except Exception:
            write_json(intent_path,{**intent,'state':'UNKNOWN'});raise
        if not jid.isdigit():write_json(intent_path,{**intent,'state':'UNKNOWN'});raise ValueError('Cannot parse submitted job ID')
        history.append(dict(job_id=jid,token=token));write_json(root/'allocations.json',history);write_json(intent_path,{**intent,'state':'REGISTERED','job_id':jid})
        (root/'job_id').write_text(jid+'\n');result=dict(state='SUBMITTED',job_id=jid,remaining_seconds=remaining,depends_on=dependencies)
        write_json(root/'submission.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.release.resolve(),a.run.resolve(),a.resume)))
