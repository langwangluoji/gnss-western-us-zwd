"""Real frozen-weight equality/timing probe; no prediction accuracy claims."""
from pathlib import Path
import sys
import time
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.native_frequency_tokens import NativeFrequencyTokens
from tools.terrain_diffusion.native_spectral_fast import CommutedFrequencyTokens
from tools.terrain_diffusion.native_weather import graph_basis
from tools.terrain_diffusion.data import sha,write_json


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.35)
    result=ENGINE/'results';folder=result/'20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens'
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    for name,expected in protocol['extra_model_sources'].items():assert sha(ENGINE/'tools/terrain_diffusion'/name)==expected
    cache=result/'20261126_native_weather_cache_v2/calendar8_s42'
    embedding=result/'20261126_native_weather_embeddings/transformer72_physics_aux_last_dev_s42'
    manifest=json.loads((embedding/'manifest.json').read_text(encoding='utf-8'))
    assert sha(embedding/'embedding.npy')==manifest['files']['embedding.npy']
    c=torch.tensor(np.load(embedding/'embedding.npy',mmap_mode='r')[0].copy(),device='cuda',dtype=torch.float32).repeat(32,1,1)
    q=np.load(cache/'query_ids.npy')[0].astype(int)
    st=torch.tensor(json.loads((cache/'geometry.json').read_text())['dev_static'],device='cuda')[q]
    eigen,U=graph_basis(st[:,:2]*1000,st[:,2]*3000)
    weights=torch.load(folder/'last.pt',weights_only=True)
    models=[NativeFrequencyTokens().cuda().eval(),CommutedFrequencyTokens().cuda().eval()]
    for model in models:model.load_state_dict(weights)
    torch.manual_seed(333);noise=torch.randn(len(c),c.shape[1],7,device='cuda')
    outputs=[];timing=[]
    for model in models:
        model.denoise(noise,c,torch.ones(len(c),device='cuda')*.5,U,eigen);torch.cuda.synchronize()
        start=time.perf_counter();sample=model.sample(c,noise,U,eigen,steps=40,solver='ddim');torch.cuda.synchronize()
        timing.append(time.perf_counter()-start);outputs.append(sample)
    scale=torch.tensor(protocol['normalization']['scale_mm'],device='cuda')
    difference=(outputs[0]-outputs[1])*scale
    maximum=float(difference.abs().max());rms=float(difference.square().mean().sqrt())
    if maximum>.005:raise AssertionError(f'backend numeric drift {maximum} mm')
    receipt=dict(full_chain_max_difference_mm=maximum,full_chain_rms_difference_mm=rms,
        original_seconds=timing[0],commuted_seconds=timing[1],observed_speed_ratio=timing[0]/timing[1],
        timing_context='One actual 12h development block, K32, DDIM40; other research CUDA jobs concurrently active. Not isolated latency.',
        state_shape=list(noise.shape),weights_sha256=sha(folder/'last.pt'),
        original_implementation_sha256=sha(ENGINE/'tools/terrain_diffusion/native_frequency_tokens.py'),
        commuted_implementation_sha256=sha(ENGINE/'tools/terrain_diffusion/native_spectral_fast.py'),
        no_modes_truncated=True,no_retraining=True,accuracy_claim=False,script_sha256=sha(Path(__file__)))
    write_json(Path(__file__).with_name('spectral_backend_benchmark.json'),receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
