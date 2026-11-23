"""A small self-contained server release; no Windows paths or old logs needed."""
from pathlib import Path
import json
import zipfile
import shutil
import numpy as np
from local_subspace_data import ROOT, PilotData
from local_probe_diagnostics import PROJECT, save_npz
from local_conditional_gaussian import wet_feature
from gnss_field.artifacts import immutable_json, write_json, sha256


def main():
    data = PilotData(device='cpu')
    c = data.arrays(2019)
    annual = dict(np.load(ROOT/'full2019/annual_means.npz'))
    cache = dict(np.load(ROOT/'full2019/A_conditions.npz'))
    np.testing.assert_array_equal(annual['hours'], cache['hours'])
    wet, present = wet_feature(data.data.development.context_values(cache['hours']))
    a = data.amplitude
    g = np.exp(a['a']+a['b']*np.where(present, (wet-a['A_wetness_center'])/a['A_wetness_std'], 0.))
    evaluation = ROOT/'server_evaluation'
    evaluation.mkdir(exist_ok=True)
    release_records = sorted((PROJECT/'results/20261119_frozen_m0_audit/release/seed_42/mean_base/full2019/5000').glob('forecast_*.npz'))
    full = np.concatenate([np.load(path)['population_F'] for path in release_records])
    save_npz(evaluation/'data.npz', hours=cache['hours'], station_ids=c['station_ids'], mean=annual['analytic_mean'],
        truth=annual['truth'], valid=annual['population'], population_F=full, condition=cache['condition'],
        amplitude=g.astype('f4'), projection=c['projection'].cpu().numpy(), remainder=c['remainder'].cpu().numpy())
    data.data.development.stations.iloc[c['station_ids']].to_csv(evaluation/'stations.csv', index=False)
    immutable_json(evaluation/'manifest.json', dict(role='C', year=2019, data_sha256=sha256(evaluation/'data.npz'),
        stations_sha256=sha256(evaluation/'stations.csv'), tau_surface_mm=float(data.sigma[-1]),
        training_bundle_manifest_sha256=sha256(ROOT/'server_bundle/manifest.json'),
        inference='A-only cached conditions, frozen B statistics; labels only for evaluator; D absent'))
    release_name = '20261120_observable_subspace_v02'
    staging = PROJECT/'build/obs64v2'
    staging.mkdir(parents=True, exist_ok=True)
    for path in (PROJECT/'src/gnss_field').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.json', '.html'):
            destination = staging/'src/gnss_field'/path.relative_to(PROJECT/'src/gnss_field')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for folder, origin, names in [('bundle', ROOT/'server_bundle', ['data.npz', 'spatial.npz', 'manifest.json', 'preparation.lock.json', 'validation.json']),
                                   ('evaluation', evaluation, ['data.npz', 'stations.csv', 'manifest.json'])]:
        (staging/folder).mkdir(exist_ok=True)
        for name in names:
            shutil.copy2(origin/name, staging/folder/name)
    (staging/'tools').mkdir(exist_ok=True)
    shutil.copy2(PROJECT/'tools/lock_observable_budget.py', staging/'tools/lock_observable_budget.py')
    (staging/'slurm').mkdir(exist_ok=True)
    for path in (PROJECT/'slurm').glob('observable_*.sh'):
        shutil.copy2(path, staging/'slurm'/path.name)
    (staging/'configs').mkdir(exist_ok=True)
    shutil.copy2(PROJECT/'configs/observable_coverage_a800.json', staging/'configs/observable_coverage_a800.json')
    shutil.copy2(PROJECT/'docs/observable_server_handoff.md', staging/'README.md')
    files = {str(path.relative_to(staging)).replace('\\', '/'): sha256(path) for path in staging.rglob('*') if path.is_file() and path.name != 'release.json'}
    write_json(staging/'release.json', dict(files=files, study='observable64_coverage_v01', no_D2020=True,
        environment='existing ZhouSiyu, no installs/upgrades', source_version='0.1.2 plus locked observable modules'))
    archive = PROJECT/'dist/20261120_observable64_v02.zip'
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in staging.rglob('*'):
            if path.is_file():
                z.write(path, str(Path(release_name)/path.relative_to(staging)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    write_json(ROOT/'server_release.json', dict(archive=str(archive), sha256=sha256(archive),
        bytes=archive.stat().st_size, release_files=len(files)+1, training_B2018_only=True,
        evaluation_C2019_separate=True, D2020_included=False, submitted=False))
    print(f'Portable archive: {archive.name}; {archive.stat().st_size/2**20:.2f} MiB; CRC passed', flush=True)


if __name__ == '__main__':
    main()
