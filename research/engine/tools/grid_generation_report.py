"""Dependency-free SVG and JSON pilot report from actual completed artifacts."""
from pathlib import Path
import json,html
import numpy as np
from gnss_field.artifacts import read_json,write_json


def color(value,low,high):
    t=float(np.clip((value-low)/max(high-low,1e-10),0,1))
    a=np.array([35,75,110]); b=np.array([245,211,110]); c=(1-t)*a+t*b
    return '#'+''.join(f'{int(v):02x}' for v in c)


def build_report(root):
    root=Path(root); results={}; blocks=[]; curves=[]
    baselines={p.parent.name:read_json(p) for p in root.glob('fold_*/gaussian_outer.json')}
    for i,path in enumerate(sorted(root.glob('fold_*/*/results.json'))):
        record=read_json(path); name=path.parent.relative_to(root).as_posix(); results[name]=record
        log=path.parent/'development.jsonl'
        rows=[json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        rows=[x for x in rows if not x['startup']]
        x0=40+(i%3)*390; y0=50+(i//3)*260
        curves.append(f'<text x="{x0}" y="{y0}">{html.escape(name)} | inner RMSE</text>')
        values=[(x['step'],x['metrics']['point']['value']) for x in rows]
        if values:
            lo=min(v for _,v in values); hi=max(v for _,v in values)
            points=' '.join(f'{x0+340*s/684:.2f},{y0+200-170*(v-lo)/max(hi-lo,.01):.2f}' for s,v in values)
            curves.append(f'<polyline points="{points}" fill="none" stroke="#246e78" stroke-width="2"/>')
            curves.append(f'<text x="{x0}" y="{y0+225}">min {lo:.4f} mm | max {hi:.4f} mm</text>')
        product=record['products']['joint']; m=product['metrics']
        base=baselines.get(path.parent.parent.name)
        if base:
            record['matched_gaussian_difference']={k:m[k]-base[k] for k in ('group_ES_mm','group_VS','CRPS_mm','station_macro_rmse_mm')}
        blocks.append(f'{name}: ES {m["group_ES_mm"]:.4f}, VS {m["group_VS"]:.4f}, CRPS {m["CRPS_mm"]:.4f}, RMSE {m["station_macro_rmse_mm"]:.4f} mm')
    header='<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="620" viewBox="0 0 1200 620"><rect width="1200" height="620" fill="#fafbfc"/><g font-family="Arial,sans-serif" font-size="13" fill="#243b4a">'
    (root/'training_curves.svg').write_text(header+''.join(curves)+'</g></svg>',encoding='utf-8')
    calibration=[]
    for i,(name,record) in enumerate(results.items()):
        m=record['products']['joint']['metrics']; x=40+(i%3)*390; y=45+(i//3)*260
        calibration.append(f'<text x="{x}" y="{y}">{html.escape(name)}</text>')
        for j,level in enumerate((80,90,95)):
            value=m[f'coverage{level}']; yy=y+30+j*55
            calibration.append(f'<text x="{x}" y="{yy}">{level}%: observed {100*value:.1f}% | width {m[f"width{level}_mm"]:.2f} mm</text><rect x="{x}" y="{yy+6}" width="{300*value:.1f}" height="10" fill="#267a80"/><line x1="{x+3*level}" x2="{x+3*level}" y1="{yy+2}" y2="{yy+20}" stroke="#b94e4e"/>')
    (root/'calibration.svg').write_text(header+''.join(calibration)+'</g></svg>',encoding='utf-8')
    for path in sorted(root.glob('fold_*/*/joint_field_case.npz')):
        a=dict(np.load(path)); pieces=[]
        for panel,key in enumerate(('mean_Z0','std_Z0')):
            field=a[key]; lo,hi=np.quantile(field[a['land'].astype(bool)],[.01,.99]); pieces.append(f'<text x="{30+panel*570}" y="30">{key}: {lo:.2f} to {hi:.2f} mm | 25km spacing</text>')
            for row in range(field.shape[0]):
                for col in range(field.shape[1]):
                    if not a['land'][row,col]: continue
                    pieces.append(f'<rect x="{30+panel*570+3.5*col:.1f}" y="{50+3.5*(field.shape[0]-1-row):.1f}" width="3.6" height="3.6" fill="{color(field[row,col],lo,hi)}"/>')
        path.with_suffix('.svg').write_text(header+''.join(pieces)+'</g></svg>',encoding='utf-8')
    write_json(root/'summary.json',dict(gaussian_baselines=baselines,experiments=results,cohort='B2018 outer96 pilot, no C/D; historical base exposure',no_final_claim=True))
    (root/'RESULTS.md').write_text('# Generation pilot results\n\nB2018 outer96 pilot; all scores on matched cohorts.\n\n'+'\n'.join('- Gaussian '+name+': RMSE '+str(m['station_macro_rmse_mm'])+' mm, group ES '+str(m['group_ES_mm']) for name,m in baselines.items())+'\n'+'\n'.join('- '+x for x in blocks)+'\n\nSee training_curves.svg, calibration.svg and fold-specific joint_field_case.svg.\n',encoding='utf-8')
