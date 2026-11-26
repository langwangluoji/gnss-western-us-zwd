"""One seed, both preregistered modes, fixed order and matched budgets."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);args=p.parse_args()
    for mode in ['factorized','joint']:
        subprocess.run([sys.executable,str(Path(__file__).with_name('directed_campaign.py')),
            '--seed',str(args.seed),'--mode',mode],check=True)


if __name__=='__main__':main()
