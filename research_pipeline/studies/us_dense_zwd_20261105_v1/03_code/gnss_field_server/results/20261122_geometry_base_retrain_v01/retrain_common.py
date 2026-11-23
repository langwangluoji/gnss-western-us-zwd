import sys,os
from pathlib import Path
PACKAGE_ROOT=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('GNSS_RETRAIN_ROOT',str(PACKAGE_ROOT))).resolve()
PROJECT=PACKAGE_ROOT.parents[1]
OLD=PROJECT/'results/20261122_local_generative_architecture_v01'
OPERATORS=PROJECT/'results/20261122_local_operator_exploration_v01'
for p in (OLD,OPERATORS):
    if str(p) not in sys.path:sys.path.insert(1,str(p))
