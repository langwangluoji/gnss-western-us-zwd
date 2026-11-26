import torch
import numpy as np


def test_real_mean_path_supports_multi_hour_gradients(legacy_fixture, tmp_path):
    from gnss_field.data import pack_legacy, PackedStore, ContextBuilder
    from gnss_field.models.mean import MeanField
    from gnss_field.training.fields import predict_patches
    pack_legacy(legacy_fixture, tmp_path/'pack', registered=False)
    store = PackedStore(tmp_path/'pack')
    ctx = ContextBuilder(store).build_context(store.hours[20:22])
    torch.manual_seed(2)
    model = MeanField().train()
    field, feat = predict_patches(model, ctx, store, chunk=1)
    assert field.shape == (2, 1, 24, 24) and feat is None
    field.square().mean().backward()
    assert model.query_head.weight.grad is not None
    assert torch.isfinite(model.query_head.weight.grad).all()


def test_training_qualification_is_reusable_and_balanced(legacy_fixture, tmp_path):
    from gnss_field.data import pack_legacy, PackedStore
    from gnss_field.data.manifests import build_training_manifest
    pack_legacy(legacy_fixture, tmp_path/'pack', registered=False)
    store = PackedStore(tmp_path/'pack', purpose='train')
    a = build_training_manifest(store, tmp_path/'qualification.npz')
    b = build_training_manifest(store, tmp_path/'qualification.npz')
    np.testing.assert_array_equal(a.valid, b.valid)
    hours = a.select([2016])
    assert len(hours) == 78
    weights = a.weights(hours)
    np.testing.assert_allclose((a.masks(hours)*weights).sum(-1).mean(), 1.)
