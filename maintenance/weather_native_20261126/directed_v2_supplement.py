"""Frozen raw/EMA matched supplement for all directed-transport arms."""
import argparse
from pathlib import Path
from campaign import RESULTS,read,run
from directed_v2_summary import locations,should_supplement
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_directed_lag_v2_experiment import code_hashes,require_export_contract
from tools.terrain_diffusion.native_increment_experiment import code_hashes as increment_codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);args=p.parse_args();seed=args.seed
    root=RESULTS/'20261126_directed_lag_v2';cache=RESULTS/f'20261126_native_weather_cache_v2/calendar15_s{seed}'
    if read(Path(__file__).with_name('directed_v2_lock.json'))['code_sha256']!=code_hashes():raise ValueError('frozen implementation changed')
    main_summary=read(root/'paired_calendar8/summary.json')
    if not should_supplement(main_summary['rows']):raise ValueError('no raw joint F6 improvement to trigger supplement')
    for arm in ['H_delta','factorized','joint']:
        for weights in ['last','ema']:
            model,embedding,evaluation=locations(RESULTS,arm,seed,weights,15)
            protocol=read(model/'protocol.json');require_export_contract(protocol,read(cache/'protocol.json'))
            expected=increment_codes() if arm=='H_delta' else code_hashes()
            if protocol['code_sha256']!=expected:raise ValueError('model implementation differs')
            module='native_increment_experiment' if arm=='H_delta' else 'native_directed_lag_v2_experiment'
            run(module,['--phase','export','--cache',cache,'--mean',model,'--weights',weights,'--out',embedding,'--seed',seed],embedding/'manifest.json')
            run(module,['--phase','evaluate','--cache',cache,'--embeddings',embedding,'--out',evaluation,'--seed',seed],evaluation/'metrics.json')
    write_json(root/f'supplement/s{seed}_complete.json',dict(complete=True,seed=seed,calendar_day=15,no_fitting=True,
        main_summary_sha256=sha(root/'paired_calendar8/summary.json'),cache_manifest_sha256=sha(cache/'manifest.json'),
        strict_oof=False,confirmation_C_D_read=False,code_sha256=sha(__file__)))
    print('Complete directed matched supplement',seed,flush=True)


if __name__=='__main__':main()
