"""Fail-closed adoption of new or reused centered-control evaluation recipes."""
import json


def validate_lineage(lineage,source,cache_sha,embedding_sha,mode,codes):
    if lineage['source']!=source or lineage['code_sha256']!=codes:raise ValueError('centered evaluation source changed')
    if lineage['cache_manifest_sha256']!=cache_sha or lineage['embedding_manifest_sha256']!=embedding_sha:
        raise ValueError('centered evaluation cache or embedding differs')
    if lineage['mode']!=mode or lineage['members']!=32 or lineage['seed']!=261126:
        raise ValueError('centered evaluation mode or member recipe differs')
    if not lineage['independent_iid_member_noise'] or lineage['query_count']!=137:raise ValueError('fixed137-query IID evaluation required')


def validate_joint(joint,lineage_sha,code_sha,increment):
    if joint['source_lineage_sha256']!=lineage_sha or joint['code_sha256']!=code_sha or joint['members']!=32:
        raise ValueError('joint evaluation source changed')
    if abs(joint['within_forecast_increment_fair_crps_mm']-increment)>1e-10:raise ValueError('joint increment score differs')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def centered_marker(folder,gp,cache,embedding,mode,model=None):
    if not folder.exists():return None
    from tools.terrain_diffusion.data import sha
    from tools.terrain_diffusion.native_model_seal import validate_model_seal,evaluation_marker
    from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
    from tools.terrain_diffusion.native_centered_shape_experiment import codes
    seal=validate_model_seal(gp,'COMPLETE_WEATHER_GP',gp_codes())
    source={**seal['source'], 'weather_gp_protocol_sha256':seal['protocol_sha256'],
        'weather_gp_weights_sha256':seal['weights_sha256'],'weather_gp_seal_sha256':sha(gp/'model_artifact_seal.json')}
    model_seal=validate_model_seal(model,'COMPLETE_CENTERED_SHAPE',codes()) if model else None
    marker=evaluation_marker(folder,model_seal)
    config=read(cache/'protocol.json')['config'];manifest=read(cache/'manifest.json')
    if manifest['protocol_sha256']!=sha(cache/'protocol.json'):raise ValueError('centered cache protocol changed')
    if config['calendar_day']!=8 or config['months']!=12 or config['seed']!=read(gp/'protocol.json')['config']['seed']:
        raise ValueError('centered main calendar recipe differs')
    cache_sha=sha(cache/'manifest.json');embedding_sha=sha(embedding/'manifest.json')
    if read(embedding/'manifest.json')['cache_manifest_sha256']!=cache_sha:raise ValueError('centered embedding cache differs')
    validate_lineage(read(marker),source,cache_sha,embedding_sha,mode,codes())
    return marker


def joint_marker(folder,joint_script):
    path=folder/'joint_metrics.json'
    if not path.exists():return None
    from tools.terrain_diffusion.data import sha
    validate_joint(read(path),sha(folder/'lineage.json'),sha(joint_script),read(folder/'metrics.json')['forecast_increment_fair_crps_mm'])
    return path
