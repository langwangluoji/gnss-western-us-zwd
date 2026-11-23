"""Install a wheel into an isolated target and verify outside the source directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wheel', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', default='local_validation/wheel_validation.json')
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='gnss_wheel_') as temporary:
        directory = Path(temporary)
        target = directory/'installed'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(target),
                        str(Path(args.wheel).resolve())], check=True)
        env = {**os.environ, 'PYTHONPATH': str(target), 'PYTHONIOENCODING': 'utf-8'}
        script = '''
import json,sys
from pathlib import Path
import gnss_field
from gnss_field.cli import settings, parser
from gnss_field.contracts import RuntimeConfig,ScienceConfig
from gnss_field.runtime import verified_release_hashes
from gnss_field.training.pipeline import TrainingPipeline
assert Path(gnss_field.__file__).is_relative_to(Path(sys.argv[1]))
runtime=RuntimeConfig(device='cpu',data_root=sys.argv[2],cache_root='cache')
hashes=verified_release_hashes(ScienceConfig(),runtime,sys.argv[3])
pipeline=TrainingPipeline(ScienceConfig(),runtime,'test_run')
assert pipeline._source_digest
page=Path(gnss_field.__file__).parent/'resources/monitor.html'
assert page.is_file()
print(json.dumps(dict(import_path=gnss_field.__file__,hashes=hashes,source_digest=pipeline._source_digest,
                     resources_present=True,outside_source_directory=True,training_started=False)))
'''
        result = subprocess.run([sys.executable, '-c', script, str(target), str(Path(args.data).resolve()), str(project)],
                                cwd=directory, env=env, check=True, capture_output=True, text=True, encoding='utf-8')
        report = json.loads(result.stdout)
        help_result = subprocess.run([sys.executable, '-m', 'gnss_field', '--help'], cwd=directory,
                                     env=env, check=True, capture_output=True, text=True, encoding='utf-8')
        assert 'campaign' in help_result.stdout and 'benchmark' in help_result.stdout
        report.update(cli_help_passed=True, isolated_install_verified=True)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
