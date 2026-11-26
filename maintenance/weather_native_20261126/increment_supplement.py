"""Frozen final3000 H/H_delta supplementary dates; no further fitting."""
import argparse
from pathlib import Path
from campaign import RESULTS,read,run
from increment_summary import locations
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_increment_experiment import code_hashes,require_export_contract
from tools.terrain_diffusion.native_mean_fusion_v2_experiment import verify_codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);args=p.parse_args()
    seed=args.seed;root=RESULTS/'20261126_within_forecast_increment';cache=RESULTS/f'20261126_native_weather_cache_v2/calendar15_s{seed}'
    if read(Path(__file__).with_name('increment_lock.json'))['code_sha256']!=code_hashes():raise ValueError('frozen implementation changed')
    # Main-calendar comparison must exist before the predeclared supplemental phase.
    main_summary=read(root/'paired_calendar8/summary.json')
    if main_summary['comparisons']['last']['primary_internal_increment_micro_rmse_mm']['delta']>=0:
        raise ValueError('predeclared supplementary trigger: no primary main-calendar improvement')
    for arm in ['H','H_delta']:
        for weights in ['last','ema']:
            model,embedding,evaluation=locations(RESULTS,arm,seed,weights,15)
            protocol=read(model/'protocol.json');require_export_contract(protocol,read(cache/'protocol.json'))
            if arm=='H':verify_codes(protocol)
            elif protocol['code_sha256']!=code_hashes():raise ValueError('new model source changed')
            module='native_increment_experiment' if arm=='H_delta' else 'native_mean_fusion_v2_experiment'
            run(module,['--phase','export','--cache',cache,'--mean',model,'--weights',weights,'--out',embedding,'--seed',seed],embedding/'manifest.json')
            run(module,['--phase','evaluate','--cache',cache,'--embeddings',embedding,'--out',evaluation,'--seed',seed],evaluation/'metrics.json')
    marker=root/f'supplement/s{seed}_complete.json'
    write_json(marker,dict(complete=True,seed=seed,calendar_day=15,no_fitting=True,
        main_summary_sha256=sha(root/'paired_calendar8/summary.json'),cache_manifest_sha256=sha(cache/'manifest.json'),
        strict_oof=False,confirmation_C_D_read=False,code_sha256=sha(__file__)))
    print('Complete matched supplement',seed,flush=True)


if __name__=='__main__':main()
