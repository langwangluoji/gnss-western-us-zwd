"""Frozen visible information and coupled GP initial paths across query populations."""
import argparse
import json
import sys
import torch
from campaign import RESULTS,ROOT,read

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.native_process_experiment import Sources,codes,restore_mm
from tools.terrain_diffusion.native_process_models import make_process_model
from tools.terrain_diffusion.native_process_ops import space_factors,generate_process
from tools.terrain_diffusion.native_paths import file_path
from tools.terrain_diffusion.data import sha,write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    p.add_argument('--mode',choices=['full','local','affine'],required=True)
    args=p.parse_args();seed=args.seed;args.device='cuda';torch.set_num_threads(3)
    torch.cuda.set_per_process_memory_fraction(.25)
    args.cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar8_s{seed}'
    args.embeddings=RESULTS/'20261126_native_weather_embeddings'/f'transformer72_physics_aux_last_dev_s{seed}'
    args.gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}';args.copula=args.gaussian/'copula.json'
    folder=RESULTS/'20261126_gp_process'/f's{seed}'/args.mode
    protocol=read(folder/'protocol.json')
    if protocol['code_sha256']!=codes() or read(folder/'status.json')['status']!='COMPLETE_PROCESS':raise ValueError('frozen process differs')
    source=Sources(args);model=make_process_model(args.mode).cuda().eval()
    if protocol['source']!=source.receipt:raise ValueError('reference changed')
    model.load_state_dict(torch.load(folder/'last.pt',weights_only=True));rows=[]
    for block in (0,6,12):
        c,gm,gs,base,query,static=source.batch(block);B,Q,_=c.shape;K=32
        torch.manual_seed(261126+block);eta=torch.randn(K*B,Q,7,device='cuda')
        As,_,_=space_factors(static,source.copula,query);initial=torch.einsum('qr,brh->bqh',As,eta)
        full=generate_process(model,args.mode,c.repeat(K,1,1),eta,static,source.copula,query_ids=query)
        for count in (1,64,128,137):
            index=torch.arange(count,device='cuda');st=static[index];ids=query[index]
            _,Ai,_=space_factors(st,source.copula,ids)
            coupled=torch.einsum('qr,brh->bqh',Ai,initial[:,index])
            # Full uses whitened common physical GP initial path; Local starts
            # directly from that common physical path. Both have correct marginals.
            result=generate_process(model,args.mode,c[:,index].repeat(K,1,1),coupled,st,source.copula,
                query_ids=ids,spatial_noise=None if args.mode=='full' else initial[:,index])
            delta=(result-full[:,index])*source.scale*gs[:,index].repeat(K,1,1)
            rows.append(dict(block=block,query_count=count,max_abs_path_difference_mm=float(delta.abs().max()),
                path_rms_difference_mm=float(delta.square().mean().sqrt())))
    payload=dict(mode=args.mode,rows=rows,members=32,origin_blocks=[0,6,12],
        interpretation='Common superset GP initial path, fixed visible information and same20-stepHeun0. Full path differences diagnose coupling sensitivity and do not alone prove a difference in marginal laws. Local/Affine paths should match to numerical precision.',
        process_weights_sha256=sha(folder/'last.pt'),source=source.receipt,code_sha256=sha(__file__),
        strict_oof=False,confirmation_C_D_read=False)
    write_json(folder/'query_audit.json',payload);print(json.dumps(payload))


if __name__=='__main__':main()
