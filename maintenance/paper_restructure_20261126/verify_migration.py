"""Verify every pre-migration protected file without interpreting observations."""
import csv
import json
import subprocess
from pathlib import Path

from .inventory import sha

ROOT=Path.cwd();HERE=ROOT/'maintenance/paper_restructure_20261126'
ENGINE='research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'


def exists(path):
    import os
    path=Path(path).absolute()
    return (Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path).exists()


def relocated(old):
    if old==ENGINE+'/pyproject.toml':
        return ROOT/'archive/legacy/engine_source/pyproject.toml'
    if old.startswith('research_pipeline/'):
        if old.startswith(ENGINE+'/'):
            tail=old[len(ENGINE)+1:];part=tail.split('/')[0]
            if part=='src':return ROOT/tail
            if part in ('tools','tests','configs','slurm','docs') or '/' not in tail:
                candidate=ROOT/'research/engine'/tail
                if candidate.exists():return candidate
        return ROOT/'artifacts/legacy_layout'/old
    if old.startswith('maintenance/') and not old.startswith('maintenance/paper_restructure_20261126/'):
        tail=old[len('maintenance/'):];parts=tail.split('/')
        if parts[0] in ('europe_transfer_20261126','europe_vmf3_20261126') and len(parts)>1 and parts[1] in ('cache','runs','probability','evaluation'):
            return ROOT/'artifacts/campaigns'/tail
        return ROOT/'research'/tail
    return ROOT/old


def run():
    before=list(csv.DictReader((HERE/'inventory_before.local.csv').open(encoding='utf-8')))
    deletes=json.loads((HERE/'delete_plan.json').read_text())['files'];deleted={x['path']:x for x in deletes}
    checked=total=0;failures=[];recoverable=[];regenerable=[]
    navigation={'docs/'+n:n for n in ('CURRENT_RESEARCH.md','EXPERIMENT_INDEX.md','GETTING_STARTED.md','PROJECT_ARCHITECTURE.md','LOCAL_ARTIFACTS.md')}
    for row in before:
        old=row['path'];expected=row['sha256']
        if old.startswith('maintenance/paper_restructure_20261126/'):
            continue # implementation files continue evolving; not scientific artifacts
        if old=='docs/superpowers/plans/2026-11-26-paper-ready-restructure.md':
            continue # this task's mutable implementation plan, created before inventory
        if any(part.endswith('.egg-info') for part in Path(old).parts):
            regenerable.append(old)
            continue # installer metadata is rebuilt by wheel generation
        if old in deleted:
            item=deleted[old];npz=relocated(item['surviving_npz'])
            assert sha(npz)==item['npz_sha256']
            assert not exists(relocated(old))
            continue
        path=relocated(old)
        candidates=[path]
        if old.endswith('/20261122_local_coverage_expansion_v01/cache/cursor.json'):
            candidates.insert(0,HERE/'archived_coverage_cursor.json')
        if old in ('.gitignore','AGENTS.md','README.md'):
            candidates.insert(0,ROOT/'archive/legacy/root_before'/old)
        if old in navigation:
            candidates.insert(0,ROOT/'archive/legacy/navigation_before'/navigation[old])
        if old.startswith(ENGINE+'/tools/') or old.startswith(ENGINE+'/tests/'):
            snapshot=ROOT/'archive/legacy/engine_source'/Path(old).name
            if snapshot.exists():candidates.insert(0,snapshot)
        if old==ENGINE+'/src/gnss_field/artifacts.py':
            candidates.insert(0,ROOT/'archive/legacy/engine_source/gnss_artifacts.py')
        if old=='maintenance/europe_vmf3_20261126/features.py':
            candidates.insert(0,ROOT/'archive/legacy/engine_source/europe_vmf3_features.py')
        matched=False
        for candidate in candidates:
            if exists(candidate) and sha(candidate)==expected:
                matched=True;break
        if not matched and old.endswith('.md'):
            historical=subprocess.run(['git','show','b728737:'+old],capture_output=True)
            if historical.returncode==0:
                import hashlib
                if hashlib.sha256(historical.stdout).hexdigest()==expected:
                    recoverable.append(old);matched=True
        if not matched: failures.append(dict(old=old,current=str(path)))
        checked+=1;total+=int(row['size'])
        if checked%2000==0:print(json.dumps(dict(checked=checked,gib=round(total/2**30,3))),flush=True)
    result=dict(status='passed' if not failures else 'failed',protected_files_verified=checked,protected_bytes=total,
                approved_duplicate_deletions=deletes,link_only_markdown_originals_in_git_tag=recoverable,
                failures=failures,regenerable_install_metadata=regenerable,confirmation_labels_parsed=False,training_started=False,
                source_inventory_sha256=sha(HERE/'inventory_before.local.csv'))
    (HERE/'integrity_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],verified=checked,bytes=total,failures=len(failures),examples=failures[:3]),ensure_ascii=True),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':run()
