set -eu
root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
src="$root/code/a800_evalfix_v04/gnss_field_server"
run="$root/runs/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02"
cd "$src"
export PYTHONPATH="$src/src" PYTHONNOUSERSITE=1
"$HOME/.conda/envs/ZhouSiyu/bin/python" - "$root" "$src" "$run" <<'PY'
import sys,io,base64,json,zipfile
from pathlib import Path
import gnss_field
root,src,run=map(Path,sys.argv[1:])
m=run/'seed_42/mean_base'
print('IMPORT',sys.executable,gnss_field.__file__)
print('DEPENDENCIES',json.dumps([str(p.relative_to(root)) for p in root.rglob('*') if p.name in ('decoder.pt','representation_gate.json','frozen.pt')]))
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for base,paths in [(run,[*m.glob('full2019/5000/*'),*m.glob('eval_005000/forecast_*.npz'),m/'frozen.json',run/'resolved.json',run/'slurm/phase_000.sh',run/'campaign.json']), (src,[*src.glob('src/**/*.py')])]:
  for p in paths:
   if p.is_file(): z.write(p,p.relative_to(base))
print('ZIP='+base64.b64encode(buf.getvalue()).decode())
PY
