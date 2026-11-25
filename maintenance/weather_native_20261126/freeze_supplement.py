"""Freeze primary spatial-dependence candidates before the additional calendar."""
from pathlib import Path
from datetime import datetime,timezone
import sys
from campaign import ROOT,RESULTS,read

sys.path.insert(0,str(ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'))
from tools.terrain_diffusion.data import sha,write_json


def main():
    out=Path(__file__).with_name('supplement_lock.json')
    if out.exists():raise FileExistsError('additional-calendar candidate lock already frozen')
    sources={}
    for seed in (42,17):
        paths={}
        for name in ('full','local'):
            folder=RESULTS/'20261126_gp_process'/f's{seed}'/name
            if read(folder/'status.json')['status']!='COMPLETE_PROCESS':raise ValueError('candidate must complete3000 blocks')
            for f in ('protocol.json','last.pt'):paths[str(folder/f)]=sha(folder/f)
        gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}'
        for f in ('protocol.json','last.pt','copula.json'):paths[str(gaussian/f)]=sha(gaussian/f)
        mean=RESULTS/'20261126_native_weather_means'/f'transformer72_physics_aux_s{seed}'
        for f in ('protocol.json','last.pt'):paths[str(mean/f)]=sha(mean/f)
        sources[str(seed)]=paths
    write_json(out,dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        primary_candidates=['reference','full','local'],weights='last',solver='heun0',steps=20,members=32,
        additional_calendar='2020 each month15-17,72continuous origin hours,3layouts,137registeredqueries',
        prespecified_spatial_pair_distance_km=[[0,50],[50,150],[150,300],[300,None]],
        primary_utility='F6 station-pair fair CRPS; also report marginal CRPS, fair energy, current precision and trajectory increments',
        exposure='Dates already examined by historical I1 model; frozen additional-calendar verification, not blind independent test',
        no_refitting_or_gate_selection_on_additional_calendar=True,strict_oof=False,confirmation_C_D_read=False,
        code_sha256=sha(Path(__file__))))
    print(str(out))


if __name__=='__main__':main()
