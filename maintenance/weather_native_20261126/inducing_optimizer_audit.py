"""Release audit: every sealed flow parameter has the mask-derived Adam state."""
from pathlib import Path
import torch
from campaign import RESULTS,read
from inducing_flow_series import ARMS
from inducing_flow_completion import AMENDMENT,validate_amendment,require_actual_completion
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_model_seal import validate_model_seal
from tools.terrain_diffusion.native_inducing_flow_experiment import codes


def main():
    amendment=validate_amendment();root=RESULTS/'20261126_inducing_latent_flow';rows=[];missing=[]
    for seed in (42,17):
        for family,domain in ARMS:
            name=f'{family}_{domain}_s{seed}';folder=root/name
            if not (folder/'model_artifact_seal.json').exists():missing.append(name);continue
            seal=validate_model_seal(folder,'COMPLETE_INDUCING_FLOW',codes())
            support=amendment['label_support'][str(seed)];require_actual_completion(folder,support)
            state=torch.load(folder/'checkpoint_3000.pt',map_location='cpu',weights_only=True)['optimizer']
            expected={p for g in state['param_groups'] for p in g['params']}
            if not expected or set(state['state'])!=expected:raise ValueError('Adam state does not cover all parameter groups')
            if any(int(s['step'])!=support['supervised_blocks'] for s in state['state'].values()):raise ValueError('Adam clock differs')
            rows.append(dict(name=name,parameter_state_count=len(expected),complete_parameter_coverage=True,
                supervised_updates=support['supervised_blocks'],chronological_blocks=3000,
                checkpoint_sha256=seal['checkpoint_sha256'],seal_sha256=sha(folder/'model_artifact_seal.json')))
    write_json(Path(__file__).with_name('inducing_optimizer_coverage_audit.json'),dict(complete=not missing,rows=rows,missing_unfinished=missing,
        amendment_sha256=sha(AMENDMENT),code_sha256=sha(__file__),confirmation_C_D_read=False))
    print(dict(complete=not missing,sealed_models_checked=len(rows),unfinished=missing),flush=True)


if __name__=='__main__':main()
