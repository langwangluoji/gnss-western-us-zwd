"""Queue the predeclared transport only after its final G265 source exists."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from campaign import ROOT,RESULTS,STOP,read


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True)
    args=p.parse_args();seed=args.seed
    gp=RESULTS/'20261126_weather_projective_gp'/f's{seed}/c265'
    print(json.dumps(dict(stage='waiting_for_complete_G265',seed=seed)),flush=True)
    while not (gp/'status.json').exists():
        if STOP.exists():raise SystemExit(75)
        time.sleep(15)
    if STOP.exists():raise SystemExit(75)
    if read(gp/'status.json')['status']!='COMPLETE_WEATHER_GP':raise ValueError('queued GP did not complete')
    folder=Path(__file__).parent
    subprocess.run([sys.executable,str(folder/'weather_transport_campaign.py'),'--seed',str(seed)],cwd=ROOT,check=True)
    for mode in ('local','affine'):
        if STOP.exists():raise SystemExit(75)
        subprocess.run([sys.executable,str(folder/'weather_gp_query_audit.py'),'--seed',str(seed),'--transport',
            str(RESULTS/'20261126_weather_transport'/f's{seed}'/mode)],cwd=ROOT,check=True)
    print(json.dumps(dict(stage='complete_weather_transport_and_real_query_audits',seed=seed)),flush=True)


if __name__=='__main__':main()
