"""Audit completed cache blocks without treating an active file as complete."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    protocol=json.loads((args.cache/'protocol.json').read_text(encoding='utf-8'))
    status=json.loads((args.cache/'status.json').read_text(encoding='utf-8'));count=status['completed']
    original=Path(protocol['config']['data']);geo=dict(np.load(original/'geometry.npz'))
    hours=np.load(original/'hours.npy');values=np.load(original/'values.npy',mmap_mode='r')
    development=bool(protocol['config']['calendar_day'])
    targets=np.load(original/('dev_values.npy' if development else 'values.npy'),mmap_mode='r')
    target_hours=np.load(original/('dev_hours.npy' if development else 'hours.npy'))
    arrays={name:np.load(args.cache/f'{name}.npy',mmap_mode='r') for name in ['origins','query_ids','context_ids','truth','history','mean']}
    checked=[];maximum_source=0.;maximum_state=0.
    old=None
    if development and protocol['config']['calendar_day']==8:
        old=np.load(ENGINE/f'results/20261125_observed_innovation_precision/I1_history_s{protocol["config"]["seed"]}/evaluation.npz')
    for i in range(count):
        origins=arrays['origins'][i];assert (np.diff(origins)==1).all()
        q=arrays['query_ids'][i].astype(int);c=arrays['context_ids'][i].astype(int)
        if not development:assert not np.isin(geo['groups'][q],geo['groups'][c]).any()
        expected=np.stack([targets[np.searchsorted(target_hours,origins+h)[:,None],q[None],0] for h in range(7)],-1)
        np.testing.assert_array_equal(expected,arrays['truth'][i])
        if i in (0,count-1):
            origin=int(np.searchsorted(hours,origins[0]));raw=values[origin-71:origin+12,c]
            expected_raw=np.where(np.isfinite(raw),(raw-[0,280,900])/[100,20,100],0)
            difference=float(np.max(np.abs(expected_raw-arrays['history'][i,:,:,:3])))
            maximum_state=max(maximum_state,difference);assert difference<.005
            np.testing.assert_array_equal(np.isfinite(raw),arrays['history'][i,:,:,3:6].astype(bool))
            checked.append(i)
        if old is not None:
            months=protocol['config']['months'];layout=i//(months*6);month=(i//6)%months;offset=i%6*12
            difference=float(np.max(np.abs(old['prediction'][layout,month,offset:offset+12]-arrays['mean'][i])))
            maximum_source=max(maximum_source,difference);assert difference<1e-3
    receipt=dict(cache_protocol_sha256=sha(args.cache/'protocol.json'),completed_blocks_audited=count,
        cache_status=status['status'],all_truth_values_exact=True,all_origins_hourly=True,
        training_group_isolation_verified=not development,raw_state_blocks_checked=checked,
        max_float16_raw_state_quantization=maximum_state,I1_reproduction_max_difference_mm=maximum_source if old is not None else None,
        original_data_contract_sha256=sha(original/'contract.json'),code_sha256=sha(Path(__file__)),
        confirmation_C_D_labels_read=False)
    write_json(args.out,receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
