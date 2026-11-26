"""Post-training mechanism checks, never used to select checkpoints."""
from pathlib import Path
import numpy as np
import torch
from pilot_data import Data,ROOT,monthly_subset
from pilot_models import make_generator
from run_pilot import ARMS,evaluate
from gnss_field.artifacts import immutable_json,read_json,write_json,sha256
from grid_generation_metrics import score_ensemble


def main():
    torch.set_num_threads(2);data=Data();records={};stability={}
    immutable_json(ROOT/'diagnostics.lock.json',dict(data=sha256(ROOT/'data.lock.json'),
        diagnostic='final raw vs selected; exact paired keyed-noise split; shuffled-member spatial dependence',
        fixed_case='first outer hour',seed=20261129,selection_affected=False,code=sha256(__file__)))
    for fold in (0,1):
        for arm,(kind,variant) in ARMS.items():
            directory=ROOT/f'fold_{fold}'/arm;state=torch.load(directory/'latest.pt',map_location=data.device,weights_only=False)
            torch.manual_seed(42);model=make_generator(kind,'plain' if variant=='pair' else variant,recompute=True).to(data.device)
            model.load_state_dict(state['model']);hours=monthly_subset(data.lock['outer_hours'])[:1]
            with torch.no_grad():
                a=data.sample(model,hours,fold,4,20261129,chunk=4)
                b=data.sample(model,hours,fold,4,20261129,chunk=1)
            error=float((a['samples']-b['samples']).abs().max())
            if error>.001:raise ValueError(f'Chunked sample identity differs for {arm}: {error}')
            summary=dict(final_raw_step=state['step'],sample_split_max_mm=error,
                raw_correction_rms_mm=float(a['delta'].square().mean().sqrt()),
                raw_correction_mean_rms_mm=float(a['delta'].mean(1).square().mean().sqrt()),
                final_gradient_norm=state['history'][-1]['gradient_norm'])
            if variant=='noise':summary['noise_gains']=model.net.noise_gain.detach().cpu().tolist()
            if variant=='attention':summary['attention_output_weight_norm']=float(model.net.attention.out.weight.norm())
            if variant=='mp':summary['MP_output_gain']=float(model.net.output.gain)
            records[f'{fold}/{arm}']=summary;del model,state;torch.cuda.empty_cache()
        for directory in sorted((ROOT/f'fold_{fold}').iterdir()):
            path=directory/'outer_predictions.npz'
            if not path.exists():continue
            a=dict(np.load(path));z=a['samples'];rng=np.random.default_rng(20261129)
            shuffled=np.empty_like(z)
            for t in range(len(z)):
                for n in range(z.shape[-1]):shuffled[t,:,n]=z[t,rng.permutation(z.shape[1]),n]
            m=score_ensemble(a['truth'],shuffled,a['mean'],a['population'],a['xy'])
            original=read_json(directory/'result.json')['outer']
            stability[f'{fold}/{directory.name}']=dict(original_ES=original['group_ES_mm'],shuffled_ES=m['group_ES_mm'],
                original_VS=original['group_VS'],shuffled_VS=m['group_VS'],marginal_CRPS_difference=m['CRPS_mm']-original['CRPS_mm'],
                interpretation='same marginal samples; station-wise member labels independently permuted; diagnostic only')
    write_json(ROOT/'trained_mechanism_checks.json',records)
    write_json(ROOT/'dependence_shuffle.json',stability)
    print('TRAINED_MECHANISM_CHECKS',records)


if __name__=='__main__':main()
