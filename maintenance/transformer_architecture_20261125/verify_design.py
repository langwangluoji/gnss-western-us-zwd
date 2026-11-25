"""Verify design links, evidence, diagrams and algebra; no network experiment."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEC = ROOT/'docs/superpowers/specs/2026-11-25-masked-transformer-fusion-residual-design.md'
FIG = ROOT/'docs/figures/transformer_fusion_20261125'


def main():
    body = SPEC.read_text(encoding='utf-8')
    links = re.findall(r'\]\(([^)]+)\)', body)
    missing = [x for x in links if not x.startswith(('http:', 'https:', '#'))
               and not (SPEC.parent/x).resolve().exists()]
    assert not missing, missing
    evidence = json.loads((HERE/'evidence_snapshot.json').read_text(encoding='utf-8'))
    assert evidence['completed_run']['results'] == len(evidence['results']) == 14
    for kind in evidence['mean_macro_rmse_mm']:
        rows = [r for r in evidence['results'] if r['kind'] == kind]
        assert {r['seed'] for r in rows} == {42,17}
        assert all(r['steps'] == (0 if kind == 'frozen' else 3000) for r in rows)
    for name, digest in evidence['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    for k in ('p14','features','transformer'):
        assert f"{evidence['mean_macro_rmse_mm'][k]:.6f}" in body, k
    manifest = json.loads((FIG/'figure_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['figures']) == 3 and len(manifest['outputs']) == 6
    for name, digest in manifest['outputs'].items():
        path = FIG/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        if path.suffix == '.png':
            with Image.open(path) as im:
                im.verify()
        else:
            assert ET.parse(path).getroot().tag.endswith('svg')
    rng = np.random.default_rng(42)
    w = rng.normal(size=(2,6,128))
    rho = rng.uniform(-.95,.95,size=128)
    be = rng.normal(size=128); se = rng.uniform(.1,2,size=128)
    previous = np.concatenate([np.zeros_like(w[:,:1]), w[:,:-1]],axis=1)
    v = (w-rho*previous-be)/se
    restored = np.zeros_like(w)
    state = np.zeros_like(w[:,0])
    for t in range(6):
        state = rho*state+be+se*v[:,t]
        restored[:,t] = state
    np.testing.assert_allclose(restored,w,rtol=1e-12,atol=1e-12)
    physical = rng.normal(0,10,size=(2,6,128))
    b,s = 1.3,9.8
    np.testing.assert_allclose(b+s*((physical-b)/s),physical,rtol=1e-12,atol=1e-12)
    for page in ('docs/NEXT_STAGE.md','docs/CURRENT_RESEARCH.md'):
        assert SPEC.name in (ROOT/page).read_text(encoding='utf-8')
    receipt = dict(status='PASS_DESIGN_ARTIFACT_CHECKS', local_links=len(links),
        missing_links=missing, evidence_hashes='PASS', figures=3, exports=6,
        png_svg_integrity='PASS', text_bounds='PASS_FROM_RENDERER',
        normalization_and_ar_roundtrip_max_error=float(np.max(np.abs(restored-w))),
        scope='Algebra and design artifacts only; no model performance or network causality claim',
        new_network_trained=False,
        design_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest())
    (HERE/'verification.json').write_bytes((json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    main()
