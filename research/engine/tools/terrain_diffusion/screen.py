"""Serial, bounded local screen; run count and updates explicit."""
import argparse
import subprocess
import sys
from pathlib import Path


def main(root,steps):
    root=Path(root)
    for seed in (42,17):
        for name,arm in [('b1','deterministic'),('b2','gaussian'),('d0','diffusion'),('d1','diffusion')]:
            out=root/f'{name}_s{seed}'
            if (out/'completion.json').exists():continue
            cmd=[sys.executable,'-m','tools.terrain_diffusion.train','--arm',arm,'--seed',str(seed),'--steps',str(steps),
                 '--validate-every',str(max(100,steps//5)),'--out',str(out)]
            if name=='d1':cmd+=['--encoder',str(root/f'b1_s{seed}'/'best.pt')]
            subprocess.run(cmd,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--steps',type=int,default=1000)
    a=p.parse_args();main(a.root,a.steps)
