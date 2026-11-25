"""Check the targeted review's source ledger and repository links."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOC = ROOT/'docs/research_notes/2026-11-25-temporal-diffusion-comparison.md'


def main():
    text = DOC.read_text(encoding='utf-8')
    ledger = json.loads((HERE/'sources.json').read_text(encoding='utf-8'))
    assert len(ledger['sources']) == ledger['primary_sources_count'] == 8
    assert len({s['id'] for s in ledger['sources']}) == 8
    for s in ledger['sources']:
        assert s['primary_url'] in text, s['id']
        assert urlparse(s['primary_url']).scheme == 'https'
        assert s['passages'] and s['verified_mechanism'] and s['transfer_boundary']
    missing = []
    local_count = 0
    for doc in (DOC, ROOT/'docs/superpowers/specs/2026-11-25-masked-transformer-fusion-residual-design.md'):
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not link.startswith(('https:', 'http:', '#')):
                local_count += 1
                if not (doc.parent/link).resolve().exists():
                    missing.append((str(doc),link))
    assert not missing, missing
    metadata = json.loads((HERE/'crossref_metadata.json').read_text(encoding='utf-8'))
    assert all(m['status'] == 'OK' for m in metadata)
    records = {m['doi']:m for m in metadata}
    assert records['10.1038/s43247-025-02042-5']['published_online']['date-parts'][0] == [2025,2,24]
    assert records['10.1038/s41586-024-08252-9']['published_online']['date-parts'][0] == [2024,12,4]
    assert records['10.1038/s41586-024-08252-9']['published_print']['date-parts'][0] == [2025,1,2]
    report = dict(status='PASS_REVIEW_ARTIFACT_CHECKS', sources=8,
        primary_sources_linked=8, recorded_queries=len(ledger['search_queries']),
        crossref_metadata_identity_and_dates='PASS', local_links_checked=local_count,
        missing_local_links=missing, review_sha256=hashlib.sha256(DOC.read_bytes()).hexdigest(),
        scope='Targeted literature comparison; no GNSS training or gain verification')
    (HERE/'verification.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
