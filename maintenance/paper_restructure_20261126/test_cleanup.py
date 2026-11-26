from pathlib import Path
import pytest

from maintenance.paper_restructure_20261126.inventory import inside, duplicate_proof


def test_scope_rejects_sibling_and_root(tmp_path):
    root = tmp_path / 'project'; root.mkdir()
    assert inside(root, root / 'cache/a.npy') == root / 'cache/a.npy'
    with pytest.raises(ValueError): inside(root, root.parent / 'other')
    with pytest.raises(ValueError): inside(root, root)


def test_duplicate_requires_equal_content_and_surviving_reference(tmp_path):
    a = tmp_path / 'a'; b = tmp_path / 'b'
    a.write_bytes(b'one'); b.write_bytes(b'two')
    with pytest.raises(ValueError): duplicate_proof(a, b)
    b.write_bytes(a.read_bytes())
    proof = duplicate_proof(a, b)
    assert proof['size'] == 3 and len(proof['sha256']) == 64
    assert proof['surviving_reference'] == str(b)
