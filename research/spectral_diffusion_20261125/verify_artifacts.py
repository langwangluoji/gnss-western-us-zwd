"""Validate source, audit and exported artifacts without claiming model gains."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOC = ROOT/'docs/research_notes/2026-11-25-spectral-diffusion-feasibility.md'
FIG = ROOT/'docs/figures/spectral_diffusion_20261125'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    body = DOC.read_text(encoding='utf-8')
    sources = json.loads((HERE/'sources.json').read_text(encoding='utf-8'))
    assert sources['source_count'] == len(sources['sources']) == 7
    assert len({s['id'] for s in sources['sources']}) == 7
    assert all(s['url'] in body for s in sources['sources'])
    missing = []
    for link in re.findall(r'\]\(([^)]+)\)',body):
        if not link.startswith(('https:','http:','#')) and not (DOC.parent/link).resolve().exists():
            missing.append(link)
    assert not missing, missing
    audit = json.loads((HERE/'transform_audit.json').read_text(encoding='utf-8'))
    assert audit['generator_sha256'] == sha(HERE/'audit_transforms.py')
    for name,digest in audit['source_hashes'].items():
        assert sha(ROOT/name) == digest, name
    t = audit['temporal']; g = audit['graph']
    assert t['complete_station_windows'] == sum(r['complete_station_windows'] for r in t['monthly'])
    assert str(t['complete_station_windows']) in body
    assert f"{100*t['pooled_low_frequency_energy_fraction']:.2f}%" in body
    assert f"{100*t['pooled_increment_low_frequency_energy_fraction']:.2f}%" in body
    assert t['fft_inverse_max_error_mm'] < 1e-10
    assert g['orthogonality_max_error'] < 1e-10
    assert g['transformed_diffusion_equivalence_max_error'] < 1e-10
    large = [r for r in audit['irregular_fourier']['rows'] if r['coefficients']==121]
    assert len(large)==3 and all(r['nullity']==39 and r['numerical_rank']==82 for r in large)
    manifest = json.loads((HERE/'diagram_manifest.json').read_text(encoding='utf-8'))
    assert manifest['generator_sha256'] == sha(HERE/'render_architecture.py')
    assert manifest['renderer_sha256'] == sha(ROOT/'maintenance/transformer_architecture_20261125/render_diagrams.py')
    for name,digest in manifest['outputs'].items():
        assert sha(FIG/name)==digest, name
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:im.verify()
    ET.parse(FIG/'joint_architecture.svg')
    metadata = json.loads((HERE/'crossref_metadata.json').read_text(encoding='utf-8'))
    assert len(metadata)==2 and all(r['status']=='OK' for r in metadata)
    for page in ('docs/NEXT_STAGE.md','docs/CURRENT_RESEARCH.md',
                 'docs/superpowers/specs/2026-11-25-masked-transformer-fusion-residual-design.md'):
        assert DOC.name in (ROOT/page).read_text(encoding='utf-8')
    result = dict(status='PASS_SPECTRAL_RESEARCH_ARTIFACT_CHECKS',source_records=7,
        recorded_searches=len(sources['queries']),local_links='PASS',source_hashes='PASS',
        transform_algebra='PASS',reported_numbers='PASS',png_svg_integrity='PASS',
        diagram_text_bounds='PASS_FROM_RENDERER',independent_read_only_design_review='COMPLETED',
        document_sha256=sha(DOC),
        scope='Existing exposed development residual diagnostics plus algebra only; no new spectral network trained',
        new_model_precision_gain_verified=False)
    (HERE/'verification.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
