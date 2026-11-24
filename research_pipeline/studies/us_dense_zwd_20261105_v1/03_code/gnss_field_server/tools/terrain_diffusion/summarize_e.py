import json,glob
from pathlib import Path
rows=[]
for p in glob.glob('results/20261124_E*_short/formal_summary.json'):
 d=json.loads(Path(p).read_text()); parts=Path(p).parent.name.split('_'); vals=[x['expanded_metrics']['macro_rmse_mm'] for x in d]
 rows.append({'experiment':parts[1],'sampling':parts[2],'seed42':vals[0],'seed17':vals[1],'mean':sum(vals)/2})
Path('results/20261124_E_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); print(rows)
