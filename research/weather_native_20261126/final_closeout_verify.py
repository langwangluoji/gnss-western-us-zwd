"""Verify the final lightweight delivery against its existing sources."""
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
RESULTS=ENGINE/'results'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    report=ROOT/'docs/research_notes/2026-11-26-final-research-summary.md'
    docs=[report,ROOT/'docs/CURRENT_RESEARCH.md',ROOT/'docs/EXPERIMENT_INDEX.md',ROOT/'README.md',ROOT/'docs/superpowers/plans/2026-11-26-final-weather-calibration.md']
    missing=[];links=0
    # Existing large historical indexes retain prior links; verify the new report
    # and final execution ledger in full, and other newly added entry links only.
    for p in docs:
        text=p.read_text(encoding='utf-8')
        if p.name in ('CURRENT_RESEARCH.md','EXPERIMENT_INDEX.md'):text=text.split('以下')[0]
        if p.name=='README.md':text='\n'.join(l for l in text.splitlines() if '阶段研究总报告' in l)
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if link.startswith(('http','#')):continue
            target=p.parent/unquote(link.split('#')[0]);links+=1
            if not target.exists():missing.append([str(p.relative_to(ROOT)),link])
    if missing:raise ValueError(missing)
    figures=ROOT/'docs/figures/final_research_20261126'
    manifest=read(figures/'manifest.json')
    for name,digest in manifest['sources'].items():assert sha(ROOT/name)==digest,name
    for name,digest in manifest['figures'].items():assert sha(figures/name)==digest,name
    point=read(RESULTS/'20261126_final_research_summary/point_metrics.json')
    tail=read(RESULTS/'20261126_weather_scale_tail/paired/summary.json')
    for r in point['rows']:
        for value in (r['main_R_F6']+r['supplement_R_F6']+[r['macro_rmse_mm'][0],r['macro_rmse_mm'][6]]):
            assert f'{value:.6f}' in report.read_text(encoding='utf-8'),(r['arm'],value)
    for name in ('fixed','constant','G_weather','G_extra','T_constant','T_weather'):
        for key in ('crps_h0','crps_h6','increment','pair_0-50_h6','joint_nll_per_dimension'):
            value=np.mean([r['records'][key]['value'] for r in tail['rows'] if r['name']==name])
            assert f'{value:.6f}' in report.read_text(encoding='utf-8'),(name,key,value)
    assert tail['matched_covariance_arrays_exact'] and tail['all_centers_truth_exact']
    light=[]
    for folder in (RESULTS/'20261126_weather_scale_tail',RESULTS/'20261126_final_research_summary',figures):
        light.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.png','.svg'))
    light.extend([report,ROOT/'maintenance/weather_native_20261126/final_point_metrics.py',ROOT/'maintenance/weather_native_20261126/final_research_figures.py',Path(__file__),ROOT/'maintenance/weather_native_20261126/final_closeout.md'])
    light.extend(docs[1:])
    for p in light:
        if p.suffix=='.json':read(p)
        assert p.stat().st_size<10_000_000,(p,p.stat().st_size)
    output=ROOT/'maintenance/weather_native_20261126/final_stage_paths.txt'
    output.write_text('\n'.join(sorted({p.relative_to(ROOT).as_posix() for p in light}))+'\n',encoding='utf-8')
    print(json.dumps(dict(links_checked=links,json_valid=True,figure_hashes_valid=True,report_tables_verified=True,lightweight_paths=len(set(light)),no_raw_arrays_or_weights=True)))


if __name__=='__main__':main()
