import json
from pathlib import Path
import subprocess

import pytest

from gnss_field.artifacts import release_digest, write_json
from gnss_field.cli import parser, _reservations
from gnss_field.monitor import collect_status, sync_snapshot


def test_release_hash_ignores_build_metadata(tmp_path):
    (tmp_path/'src/pkg').mkdir(parents=True)
    (tmp_path/'src/pkg/core.py').write_text('x=1')
    before = release_digest(tmp_path)
    (tmp_path/'src/pkg.egg-info').mkdir()
    (tmp_path/'src/pkg.egg-info/PKG-INFO').write_text('generated')
    assert release_digest(tmp_path) == before
    (tmp_path/'src/pkg/core.py').write_text('x=2')
    assert release_digest(tmp_path) != before


def test_reservations_require_positive_finite_explicit_values(tmp_path):
    path = tmp_path/'costs.json'
    write_json(path, {'edm': 1.5})
    assert _reservations(path) == {'edm': 1.5}
    for value in (0, -1, 'bad'):
        write_json(path, {'edm': value})
        with pytest.raises(ValueError):
            _reservations(path)
    assert parser().parse_args(['campaign', 'plan', '--reservations', str(path)]).reservations == str(path)


def test_ssh_sync_readonly_literal_path_and_stale_failure(tmp_path, monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, json.dumps({'runs': [], 'read_only': True}), '')
    monkeypatch.setattr(subprocess, 'run', run)
    remote = "/data/a b/'; touch nope; '"
    result = sync_snapshot(tmp_path, 'user@cluster', remote)
    assert result['sync_ok']
    assert calls[0][0][:4] == ['ssh', '-o', 'BatchMode=yes', '-o']
    assert 'shell' not in calls[0][1]
    assert collect_status(tmp_path)['read_only']
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 255, '', 'offline'))
    failed = sync_snapshot(tmp_path, 'user@cluster', remote)
    assert not failed['sync_ok']
    assert failed['last_success'] == result['last_success']
    assert failed['last_attempt'] >= result['last_attempt']
    assert collect_status(tmp_path)['ssh_sync'] == failed
    assert collect_status(tmp_path)['runs'] == []
    with pytest.raises(ValueError):
        sync_snapshot(tmp_path, '-oProxyCommand=bad', remote)


def test_monitor_skips_partial_rows_and_does_not_invent_metrics(tmp_path):
    folder = tmp_path/'edm'
    folder.mkdir()
    (folder/'training.jsonl').write_text('{"step":100,"loss":2}\n{"step":')
    state = collect_status(tmp_path)
    assert state['runs'][0]['training'] == [{'step':100, 'loss':2}]
    assert state['runs'][0]['development'] == []
    assert state['no_interpolated_metrics']


def test_monitor_full_year_table_reads_only_completed_rerank(tmp_path):
    seed = tmp_path/'seed_42'
    seed.mkdir()
    write_json(seed/'full2019.json', {'records': [dict(model='edm', step=2000,
        checkpoint='best.pt', year=2019, full_year=True, members=64,
        selection_values={'point': {'value': 7.5}, 'joint': {'value': 4.0}})]})
    result = collect_status(tmp_path)
    assert result['full2019'][0]['model'] == 'edm'
    assert result['full2019'][0]['selection_values']['point']['value'] == 7.5
    assert result['runs'] == []  # no fabricated training observations


def test_cli_bootstrap_dry_run_status_does_not_call_slurm(legacy_fixture, tmp_path, monkeypatch, capsys):
    from gnss_field.data import pack_legacy
    from gnss_field.cli import main
    packed = tmp_path/'packed'
    pack_legacy(legacy_fixture, packed, registered=False)
    def forbidden(*a, **k):
        raise AssertionError('Read-only planning must not call Slurm')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    root = tmp_path/'campaign'
    args = ['--data-root', str(packed), 'campaign']
    assert main(args+['plan', '--scope', 'bootstrap', '--directory', str(root)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['tasks'] == 4 and plan['worst_case_card_hours'] == 4
    assert main(args+['status', '--directory', str(root)]) == 0
    capsys.readouterr()
    assert main(args+['pause', '--directory', str(root)]) == 0
    assert json.loads(capsys.readouterr().out)['dry_run']
    assert not (root/'PAUSE').exists()


@pytest.mark.parametrize('changed', ['plan', 'data', 'code', 'protocol', 'synthetic'])
def test_benchmark_cli_rejects_changed_release_before_training(tmp_path, monkeypatch, changed):
    from dataclasses import asdict
    from gnss_field.cli import main
    from gnss_field.contracts import ScienceConfig, RuntimeConfig
    from gnss_field.artifacts import fingerprint
    import gnss_field.runtime as runtime_module
    root = tmp_path/'campaign'
    science = ScienceConfig()
    runtime = RuntimeConfig(data_root=str(tmp_path/'data'))
    write_json(root/'resolved.json', dict(science=asdict(science), runtime=asdict(runtime),
                                         campaign={'project_root': str(tmp_path)}))
    hashes = dict(plan=fingerprint(science), data='data', code='code', protocol='protocol')
    write_json(root/'campaign.json', {'hashes': hashes})
    write_json(root/'preflight.json', {'verified': True, 'hashes': hashes})
    def verify(*args, **kwargs):
        if changed == 'synthetic':
            raise ValueError('synthetic pack')
        return {**hashes, changed: 'changed'}
    monkeypatch.setattr(runtime_module, 'verified_release_hashes', verify)
    monkeypatch.setattr(runtime_module, 'benchmark', lambda *a, **k: pytest.fail('training reached'))
    assert main(['benchmark', '--campaign-dir', str(root)]) == 1


def test_monitor_page_renders_sync_status_and_preserves_metrics():
    page = (Path(__file__).parents[1]/'src/gnss_field/resources/monitor.html').read_text(encoding='utf-8')
    assert 'payload.ssh_sync' in page
    assert 'last_success' in page and 'last_attempt' in page
    assert 'sync_ok' in page and 'sync.error' in page
    assert page.index('payload.ssh_sync') < page.index('if(!r)return')


@pytest.mark.parametrize('override', ['config', 'data-root'])
def test_benchmark_rejects_explicit_override_before_training(tmp_path, monkeypatch, override):
    from dataclasses import asdict
    from gnss_field.cli import main
    from gnss_field.contracts import ScienceConfig, RuntimeConfig
    import gnss_field.runtime as runtime_module
    root = tmp_path/'campaign'
    config = dict(science=asdict(ScienceConfig()), runtime=asdict(RuntimeConfig()),
                  campaign={'project_root': str(tmp_path)})
    write_json(root/'resolved.json', config)
    config['science']['rank'] = 64
    caller = tmp_path/'caller.json'
    write_json(caller, config)
    monkeypatch.setattr(runtime_module, 'benchmark', lambda *a, **k: pytest.fail('training reached'))
    value = str(caller) if override == 'config' else str(tmp_path/'other-data')
    assert main(['--'+override, value, 'benchmark', '--campaign-dir', str(root)]) == 1



def test_queue_status_is_collected_read_only(tmp_path):
    manifest = {'campaign_id': 'test', 'state': 'SUBMITTED', 'jobs': [{'job_id': '1'}],
                'reserved_card_hours': 2}
    write_json(tmp_path/'manifest.json', manifest)
    assert collect_status(tmp_path)['campaign_queue'] == manifest
    assert json.loads((tmp_path/'manifest.json').read_text()) == manifest


def test_monitor_javascript_shows_disconnect_without_changing_metrics():
    import shutil
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the frontend execution check')
    page = (Path(__file__).parents[1]/'src/gnss_field/resources/monitor.html').read_text(encoding='utf-8')
    script = page.split('<script>')[1].split('</script>')[0]
    script = script.replace('refresh();setInterval(refresh,5000);', '')
    harness = r"""
const nodes = new Map();
global.document = {getElementById(id) {
  if (!nodes.has(id)) nodes.set(id, {textContent:'', value:'', innerHTML:'',
    setAttribute(){}, replaceChildren(){}, append(){}});
  return nodes.get(id);
}, createElement(){ return {append(){},textContent:''}; }};
"""
    checks = r"""
payload={runs:[{name:'edm',status:{stage:'edm',status:'RUNNING',step:100,updated:100},
 training:[{step:100,loss:2}],development:[]}],ssh_sync:{sync_ok:true,last_success:101,last_attempt:101}};
render();
const previous = ['state','updated','detail','loss'].map(id=>JSON.stringify(nodes.get(id)));
if (!nodes.get('sync').textContent.includes('SSH同步成功')) throw Error('success missing');
payload.ssh_sync={sync_ok:false,last_success:101,last_attempt:102,error:'offline'};
render();
if (!nodes.get('sync').textContent.includes('SSH同步失败') || !nodes.get('sync').textContent.includes('offline')) throw Error('failure missing');
if (JSON.stringify(previous)!==JSON.stringify(['state','updated','detail','loss'].map(id=>JSON.stringify(nodes.get(id))))) throw Error('metrics changed');
"""
    result = subprocess.run([node, '-e', harness+script+checks], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
