"""Reconstruct fitted coefficients and freeze the primary choice before new dates."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import torch

from precision_analysis import ENGINE
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.innovation_refinement import solve_ridge
from tools.terrain_diffusion.innovation_experiment import CASES


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(3)
    destination=args.experiment/'frozen_selection.json'
    if destination.exists():raise FileExistsError('selection lock already exists')
    protocol=json.loads((args.experiment/'protocol.json').read_text(encoding='utf-8'))
    comparison=json.loads(args.comparison.read_text(encoding='utf-8'))
    gates={r['case']:r for r in comparison['candidate_gates']}
    if not all(name in gates for name in CASES):raise ValueError('primary comparison lacks completed paired heads')
    coefficients={};checks=[]
    for seed in (42,17):
        path=args.experiment/f'trajectory_s{seed}/statistics_3000.pt'
        state=torch.load(path,weights_only=True,map_location='cpu')
        if state['step']!=3000 or state['seed']!=seed or state['data_sha256']!=protocol['data_sha256'] or state['source_sha256']!=protocol['source_weights'][str(seed)]:raise ValueError('statistics lineage differs')
        for name,d in CASES.items():
            fitted=args.experiment/f'{name}_s{seed}/coefficients.pt'
            actual=torch.load(fitted,weights_only=True,map_location='cpu')
            derived=solve_ridge(state['matrix'][:d,:d],state['rhs'][:d])
            if not torch.equal(actual,derived):raise ValueError('saved coefficients do not reproduce from final statistics')
            coefficients[f'{name}_s{seed}']=sha(fitted)
            checks.append(dict(case=name,seed=seed,statistics_sha256=sha(path),coefficients_exact=True,max_difference=0.))
    choice='reference'
    if gates['I1_history']['eligible_for_next_validation']:choice='I1_history'
    elif gates['I2_seasonal']['eligible_for_next_validation']:choice='I2_seasonal'
    elif gates['I0_current']['eligible_for_next_validation']:choice='I0_current'
    direct=next((r for r in comparison['direct_representation_contrasts']
                 if r['candidate']=='I2_seasonal' and r['control']=='I1_history'),None)
    if choice=='I1_history' and gates['I2_seasonal']['eligible_for_next_validation'] and direct and direct['improves_both_seeds'] and direct['negative_interval_upper']:
        choice='I2_seasonal'
    write_json(destination,dict(created_utc=datetime.now(timezone.utc).isoformat(),primary_candidate=choice,
        rule='Require paired point/seed/increment gates; prefer I1 unless I2 has supported extra paired gain',
        primary_comparison_sha256=sha(args.comparison),training_protocol_sha256=sha(args.experiment/'protocol.json'),
        training_summary_sha256=sha(args.experiment/'summary.json'),coefficient_sha256=coefficients,
        checks=checks,additional_calendar_read_by_this_freezer=False,code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(primary_candidate=choice,exact_coefficient_checks=len(checks))))


if __name__=='__main__':main()
