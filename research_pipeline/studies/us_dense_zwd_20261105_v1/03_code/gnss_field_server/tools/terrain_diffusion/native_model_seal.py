"""Final-checkpoint release receipts and complete evaluation artifacts."""
import json
from pathlib import Path
import torch
from .data import sha,write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def validate_model_seal(folder,expected_status,expected_codes):
    seal=read(folder/'model_artifact_seal.json');status=read(folder/'status.json');protocol=read(folder/'protocol.json')
    if status['status']!=expected_status or status['steps']!=3000:raise ValueError('completed model status changed')
    if protocol['code_sha256']!=expected_codes or seal['code_sha256']!=expected_codes:raise ValueError('model implementation changed')
    for name,key in (('last.pt','weights_sha256'),('checkpoint_3000.pt','checkpoint_sha256'),('protocol.json','protocol_sha256'),('status.json','status_sha256')):
        if sha(folder/name)!=seal[key]:raise ValueError('sealed model artifact changed: '+name)
    return seal


def seal_completed_model(folder,expected_status,expected_codes):
    path=folder/'model_artifact_seal.json'
    if path.exists():return validate_model_seal(folder,expected_status,expected_codes)
    status=read(folder/'status.json');protocol=read(folder/'protocol.json')
    if status['status']!=expected_status or status['steps']!=3000 or protocol['code_sha256']!=expected_codes:
        raise ValueError('cannot seal unfinished or changed implementation')
    final=torch.load(folder/'checkpoint_3000.pt',map_location='cpu',weights_only=True)
    weights=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    if final['step']!=3000 or final['model'].keys()!=weights.keys() or any(not torch.equal(final['model'][k],v) for k,v in weights.items()):
        raise ValueError('last weights differ from final checkpoint')
    result=dict(weights_sha256=sha(folder/'last.pt'),checkpoint_sha256=sha(folder/'checkpoint_3000.pt'),
        protocol_sha256=sha(folder/'protocol.json'),status_sha256=sha(folder/'status.json'),code_sha256=expected_codes,
        source=protocol['source'],seal_implementation_sha256=sha(Path(__file__)),final_step=3000,
        contract='last.pt equals checkpoint3000.model tensor by tensor; immutable completion artifact receipt')
    write_json(path,result);return result


def evaluation_marker(folder,seal=None):
    if not folder.exists():return None
    if not all((folder/n).exists() for n in ('evaluation.npz','metrics.json','lineage.json')):
        raise ValueError('incomplete evaluation directory; preserve it and restart in a new directory: '+str(folder))
    lineage=read(folder/'lineage.json');read(folder/'metrics.json')
    if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('evaluation array changed')
    if seal is not None:
        protocol=lineage.get('model_protocol_sha256',lineage.get('process_protocol_sha256'))
        if lineage['probability_weights_sha256']!=seal['weights_sha256'] or protocol!=seal['protocol_sha256']:
            raise ValueError('evaluation does not use sealed model')
    return folder/'lineage.json'
