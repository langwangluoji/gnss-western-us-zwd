"""Apply the preregistered paired A gate, then execute bounded B runs."""
import json
import subprocess
import sys
import numpy as np
from .transport_train import ROOT
from .data import write_json


def main():
    scores={}
    for variant in ('base','raw','transport'):
        scores[variant]=[]
        for seed in (42,17):
            run=ROOT/f'A_{variant}_s{seed}'
            if not (run/'completion.json').exists():raise RuntimeError('A stage must finish before gate')
            scores[variant].append(json.loads((run/'metrics.json').read_text())['macro_rmse_mm'])
    eligible=[v for v in ('raw','transport') if all(x<y for x,y in zip(scores[v],scores['base']))]
    selected=min(eligible,key=lambda v:np.mean(scores[v])) if eligible else None
    write_json(ROOT/'A_gate.json',dict(selection_scores=scores,eligible_variants=eligible,selected_variant=selected,
        rule='Improve both seeds over paired base; choose single variant by two-seed mean quarterly-screen RMSE',
        full_calendar_used_for_selection=False))
    if selected is None:print('A gate not passed; no B training');return
    for seed in (42,17):
        for arm in ('gaussian','diffusion'):
            out=ROOT/f'B_{arm}_s{seed}'
            if (out/'completion.json').exists():continue
            subprocess.run([sys.executable,'-m','tools.terrain_diffusion.transport_train','--variant',selected,'--arm',arm,
                '--seed',str(seed),'--steps','3000','--encoder',str(ROOT/f'A_{selected}_s{seed}'/'best.pt'),'--out',str(out)],check=True)


if __name__=='__main__':main()
