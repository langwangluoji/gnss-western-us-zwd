"""E1305: user-requested larger U-Net, same inputs/data/budget and native grid readout."""
import argparse,json,time,ctypes
from pathlib import Path
import torch
from .unet_foundation import Experiment,ROOT
from .data import write_json,sha

def run():
    p=argparse.ArgumentParser();p.add_argument('--budget',type=float,default=600);p.add_argument('--smoke',action='store_true');args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    assert json.loads((ROOT/'E1304_point_unet/status.json').read_text())['state']=='completed','Wait for the preceding hybrid experiments'
    root=ROOT/('large_smoke' if args.smoke else 'E1305_unet48');root.mkdir(exist_ok=False);ex=Experiment();start=time.monotonic()
    if args.smoke:ex.monitor=ex.monitor[:2]
    write_json(root/'protocol.json',dict(user_requested_width_increase=True,widths=[48,96,192],parameters=1086673,budget_seconds=args.budget,smoke=args.smoke,inputs_and_events='Same113 features,3456events/2880fit/96monitor,original validation calendar',events_sha256=sha(ROOT/'events.npz'),source_sha256=sha(__file__),foundation_source_sha256=sha(Path(__file__).with_name('unet_foundation.py')),sampler='Corrected epoch boundary; prior primary weights and executed source remain preserved'))
    for seed in ([42] if args.smoke else [42,17]):
        if not args.smoke:write_json(ROOT/'overall_status.json',dict(state='training_E1305',seed=seed,previous_runs_completed=8))
        folder=root/f's{seed}';write_json(root/'status.json',dict(state='training',seed=seed));model=ex.fit_model('unet',48,seed,folder,args.budget,5)
        if args.smoke:
            e=dict(row=0,hour=0,context=ex.data.layouts()[0]);pred=ex.predict(model,e,'unet',True);assert torch.isfinite(pred).all();metrics=dict(finite=True)
        else:metrics=ex.evaluate(model,'unet',folder)
        write_json(folder/'completion.json',dict(status='completed',best_sha256=sha(folder/'best.pt'),metrics=metrics));del model;torch.cuda.empty_cache()
    write_json(root/'status.json',dict(state='completed',seconds=time.monotonic()-start))
    if not args.smoke:write_json(ROOT/'overall_status.json',dict(state='completed',training_runs=10,primary_status='status.json',extensions=['E1304_point_unet/status.json','E1305_unet48/status.json']))

if __name__=='__main__':run()
