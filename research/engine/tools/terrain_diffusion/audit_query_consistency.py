import json
import pathlib
import torch
from .fixed_support import FixedSupportModel
from .fixed_support_experiment import support_for, uv_for
from .unet_foundation import Experiment

ex = Experiment()
weight = pathlib.Path('E:/Document/FILE_package/Research/SITP/基于生成对抗网络(GAN)的高精度对流层构建/research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_fixed_support_encoder_p14_formal2/s42/best.pt')
model = FixedSupportModel().cuda(); model.load_state_dict(torch.load(weight, weights_only=True)); model.eval()
event = dict(row=0, hour=0, context=ex.data.layouts()[0].tolist())
support = support_for(ex, event, True); geo = ex.val_features.geo; point = ex.get_features(event, 'mlp', True)
with torch.no_grad():
    full = model(point, support, uv_for(ex, geo)); first = model(point[:64], support, uv_for(ex, geo[:64]))
d = (full[:64] - first).abs()
out = {'rms_first64_vs_137': float(torch.sqrt((d*d).mean())), 'max_first64_vs_137': float(d.max()), 'finite': bool(torch.isfinite(full).all())}
pathlib.Path('results/20261124_optimization_campaign_p15/query_consistency.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(out)
