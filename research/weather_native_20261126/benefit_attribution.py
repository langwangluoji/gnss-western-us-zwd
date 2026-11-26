"""Reproduce the spatial benefit table and its nonlinear Gaussian contrast."""
from pathlib import Path
import sys
import numpy as np
from campaign import ROOT,RESULTS,read
from process_summary import scalar_interval

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json


def main():
    rows=[]
    for seed in (42,17):
        base=RESULTS/'20261126_gp_process'/f's{seed}'
        folders={m:base/m/('eval_heun0' if m!='reference' else '') for m in ('reference','full','local','affine','graph_affine')}
        scores={m:read(p/'joint_metrics.json') for m,p in folders.items()}
        ref=scores['reference'];full=scores['full'];local=scores['local'];ga=scores['graph_affine']
        record=dict(seed=seed,Full_F6_station_pair={},Full_current_station_pair={},
            Local_increment_gain_percent=100*(ref['within_forecast_increment_fair_crps_mm']-local['within_forecast_increment_fair_crps_mm'])/ref['within_forecast_increment_fair_crps_mm'],
            source_joint_sha256={m:sha(p/'joint_metrics.json') for m,p in folders.items()})
        for distance in ('0-50','50-150','150-300','300-inf'):
            for lead,key in [('h6','Full_F6_station_pair'),('h0','Full_current_station_pair')]:
                a=ref['spatial_pair_crps'][distance][lead]['fair_crps_mm'];b=full['spatial_pair_crps'][distance][lead]['fair_crps_mm']
                record[key][distance]=dict(reference_mm=a,Full_mm=b,gain_percent=100*(a-b)/a)
        a=full['spatial_pair_crps']['0-50']['h6'];b=ga['spatial_pair_crps']['0-50']['h6']
        record['Full_vs_GraphAffine_near_F6']=dict(gain_percent=100*(b['fair_crps_mm']-a['fair_crps_mm'])/b['fair_crps_mm'],
            delta_ci95_mm=scalar_interval(np.asarray(a['monthly_score_sum']),np.asarray(a['monthly_valid_count']),np.asarray(b['monthly_score_sum']),np.asarray(b['monthly_valid_count'])))
        rows.append(record)
    out=RESULTS/'20261126_diffusion_mean_sampling_audit';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'benefit_attribution.json',dict(rows=rows,code_sha256=sha(__file__),
        scope='Exposed2020 main calendar; common-month exploratory intervals; no multiplicity correction',strict_oof=False,confirmation_C_D_read=False))


if __name__=='__main__':main()
