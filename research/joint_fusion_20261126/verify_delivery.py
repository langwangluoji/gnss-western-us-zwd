"""Verify the complete fusion delivery and produce an explicit lightweight stage list."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
import numpy as np
import torch
from campaign import ROOT,HERE,OUTPUT,LOCK,read,paths,validate_lock,verify_training
from tools.terrain_diffusion.data import sha,write_json


def main():
    torch.set_num_threads(2);validate_lock();model_receipts={}
    for seed in (42,17):
        validate_lock(seed)
        for kind,folder in [('gaussian','S175'),('gp','G265'),('affine','affine'),('local','local')]:
            target=OUTPUT/f's{seed}/{folder}';receipt=verify_training(target,seed,kind)
            state=torch.load(target/'checkpoint_3000.pt',map_location='cpu',weights_only=True)
            ids={i for group in state['optimizer']['param_groups'] for i in group['params']}
            if set(state['optimizer']['state'])!=ids:raise ValueError('some registered optimizer parameter lacks complete state')
            model_receipts[f'{seed}/{kind}']=receipt
    summary=read(OUTPUT/'paired/summary.json');diagnostics=read(OUTPUT/'paired/diagnostics.json')
    if len(summary['rows'])!=24 or len(diagnostics['rows'])!=24:raise ValueError('missing calendar/seed/model output')
    if not summary['all_point_fields_equal_frozen_joint'] or not diagnostics['original_point_arrays_exact']:raise ValueError('point source not verified')
    lookup={(r['day'],r['seed'],r['name']):r for r in summary['rows']}
    for seed in (42,17):
        for day in (8,15):
            ref=lookup[day,seed,'local']['point_R_F6_mm']
            for name in ('G265','affine','local','paired_G265','paired_affine','paired_local'):
                np.testing.assert_array_equal(lookup[day,seed,name]['point_R_F6_mm'],ref)
    figure=ROOT/'docs/figures/best_base_diffusion_fusion_20261126';manifest=read(figure/'manifest.json')
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('figure source changed: '+name)
    for name,digest in manifest['figures'].items():
        if sha(figure/name)!=digest:raise ValueError('figure changed: '+name)
    docs=[ROOT/'docs/research_notes/2026-11-26-best-base-diffusion-fusion.md',
        ROOT/'docs/superpowers/specs/2026-11-26-best-base-diffusion-fusion.md',
        ROOT/'docs/superpowers/plans/2026-11-26-best-base-diffusion-fusion.md']
    links=0
    for p in [*docs,OUTPUT/'paired/REPORT.md',OUTPUT/'paired/DIAGNOSTICS.md']:
        for href in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if href.startswith(('http','#')):continue
            links+=1
            if not (p.parent/unquote(href.split('#')[0])).exists():raise ValueError('broken delivery link: '+href)
    report=dict(lock_sha256=sha(LOCK),model_receipts=model_receipts,point_arrays_preserved=True,
        row_count=24,figure_source_hashes_valid=True,links_checked=links,
        summary_sha256=sha(OUTPUT/'paired/summary.json'),diagnostics_sha256=sha(OUTPUT/'paired/diagnostics.json'),
        code_sha256=sha(__file__),strict_oof=False,confirmation_C_D_read=False)
    write_json(HERE/'delivery_verification.json',report)
    files=[]
    for folder in (HERE,OUTPUT,figure):
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.md','.png','.svg'):
                if p.stat().st_size>10_000_000:raise ValueError('non-lightweight staged artifact: '+str(p))
                if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8'))
                files.append(p)
    files+=docs+[ROOT/'docs/CURRENT_RESEARCH.md',ROOT/'docs/EXPERIMENT_INDEX.md',ROOT/'README.md']
    # OUTPUT can carry the Windows native prefix; resolve all relative spellings
    # against its known repository suffix rather than shell-built file operations.
    def relative(p):
        value=str(p)
        if value.startswith('\\\\?\\'):value=value[4:]
        return Path(value).relative_to(ROOT).as_posix()
    stages=sorted({relative(p) for p in files})
    (HERE/'stage_paths.txt').write_text('\n'.join(stages)+'\n',encoding='utf-8')
    print(json.dumps(dict(models_verified=len(model_receipts),outputs_verified=24,links_checked=links,stage_paths=len(stages))),flush=True)


if __name__=='__main__':main()
