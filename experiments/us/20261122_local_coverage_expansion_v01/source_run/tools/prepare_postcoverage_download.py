"""Read-only SSH export of two selected checkpoints and small audit files."""
import json
from pathlib import Path


def main():
    p = Path(__file__).resolve().parents[1]
    mean = json.loads((p/'results/20261119_local_edm_subspace_v01/server_results_20261120.json').read_text(encoding='utf-8'))
    edm = json.loads((p/'results/20261120_edm_coverage_cf01/server_results.json').read_text(encoding='utf-8'))
    hashes = dict(mean=mean['selected']['checkpoint_sha256'], edm=edm['selected']['checkpoint_sha256'])
    local = p/'results/20261120_postcoverage_analysis_v01'
    local.mkdir(exist_ok=True)
    (local/'expected.json').write_text(json.dumps(dict(checkpoints=hashes,D2020=False,read_only=True),indent=2),encoding='utf-8')
    folder = p/'dist/20261120_postcoverage_download'
    folder.mkdir(exist_ok=True)
    command = '''set -euo pipefail
"$HOME/.conda/envs/ZhouSiyu/bin/python" - <<'PY_EXPORT'
import base64,hashlib,io,json,zipfile
from pathlib import Path
r=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/runs/20261120_us_dense_zwd_observable64_coverage_v02'
expected=''' + repr(hashes) + '''
buffer=io.BytesIO(); files={}
with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for arm in ('mean','edm'):
  selected=json.loads((r/f'{arm}_seed42/selection.lock.json').read_text())
  checkpoint=Path(selected['checkpoint']); data=checkpoint.read_bytes()
  assert hashlib.sha256(data).hexdigest()==expected[arm]==selected['checkpoint_sha256']
  name=f'{arm}/selected.pt'; z.writestr(name,data); files[name]=expected[arm]
  for source in ['selection.lock.json','status.json','development.jsonl','training.jsonl','contract.json']:
   path=r/f'{arm}_seed42'/source
   if path.is_file():
    name=f'{arm}/{source}'; data=path.read_bytes(); z.writestr(name,data); files[name]=hashlib.sha256(data).hexdigest()
  for source in ['metrics.json','status.json','evaluation.lock.json']:
   path=r/f'{arm}_full2019'/source
   if path.is_file():
    name=f'{arm}/C2019_{source}'; data=path.read_bytes(); z.writestr(name,data); files[name]=hashlib.sha256(data).hexdigest()
 z.writestr('manifest.json',json.dumps(dict(files=files,D2020=False,read_only=True)))
data=buffer.getvalue(); assert len(data)<20*1024*1024
print('EXPORT_SHA256',hashlib.sha256(data).hexdigest(),flush=True)
print('BEGIN_CHECKPOINT_EXPORT',flush=True)
print(base64.b64encode(data).decode('ascii'),flush=True)
print('END_CHECKPOINT_EXPORT',flush=True)
PY_EXPORT
'''
    (folder/'download.stdin').write_text(command,encoding='utf-8',newline='\n')
    wrapper=('@echo off\r\ntitle GNSS postcoverage read-only checkpoint download\r\n'
        '"%SystemRoot%\\System32\\OpenSSH\\ssh.exe" -T -o ConnectTimeout=15 -o StrictHostKeyChecking=yes '
        '-o ServerAliveInterval=30 -p 10022 u23114@logini.tongji.edu.cn "bash -s" '
        '< "%~dp0download.stdin" > "%~dp0download.log"\r\n'
        'echo Read-only export finished with exit code %ERRORLEVEL%.\r\npause\r\n')
    (folder/'download.cmd').write_text(wrapper,encoding='ascii',newline='')
    print('Prepared read-only selected-checkpoint download; no server job or environment change.')


if __name__ == '__main__':
    main()
