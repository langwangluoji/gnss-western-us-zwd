import unittest
import torch
from residual_edm_v1.model import ResidualModel, edm_coefficients, keyed_noise, heun_sample

torch.set_num_threads(2)


def example(batch_size=2):
    torch.manual_seed(7)
    domain = torch.zeros(1, 1, 24, 24, dtype=torch.bool)
    domain[..., 3:21, 4:20] = True
    stations = torch.randn(batch_size, 39, 17)
    padding = torch.zeros(batch_size, 39, dtype=torch.bool)
    padding[:, -5:] = True
    return dict(coarse=torch.randn(batch_size, 4, 15, 6, 6), stations=stations,
                padding=padding, maps=torch.randn(batch_size, 6, 24, 24),
                mu_condition=torch.randn(batch_size, 1, 24, 24),
                baseline_condition=torch.randn(batch_size, 1, 24, 24),
                domain=domain, residual=torch.randn(batch_size, 1, 24, 24)*domain)


class ModelTests(unittest.TestCase):
    def test_coefficients(self):
        sigma = torch.tensor([.5, 1.])
        skip, out, inp, noise, weight = edm_coefficients(sigma)
        torch.testing.assert_close(skip, .25/(sigma.square()+.25))
        torch.testing.assert_close(out, sigma*.5/(sigma.square()+.25).sqrt())
        torch.testing.assert_close(inp, (sigma.square()+.25).rsqrt())
        torch.testing.assert_close(noise, sigma.log()/4)
        torch.testing.assert_close(weight, (sigma.square()+.25)/(sigma*.5).square())

    def test_keyed_noise_prefix_and_chunks(self):
        a = keyed_noise([100, 101], 4, 42, (1, 3, 3), device='cpu')
        b = keyed_noise([101], 2, 42, (1, 3, 3), device='cpu')
        torch.testing.assert_close(a[1, :2], b[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[0, 0], a[0, 1]))

    def test_heun_exact_constant_and_domain(self):
        domain = torch.tensor([[[[1., 0.], [1., 0.]]]])
        initial = torch.randn_like(domain)
        target = torch.full_like(domain, .3)*domain
        result = heun_sample(lambda x, s: target.expand_as(x), initial, domain, steps=4)
        torch.testing.assert_close(result, target, atol=1e-6, rtol=0)

    def test_shapes_losses_and_gradients(self):
        batch = example(1)
        for arm in ['det', 'gaussian', 'edm']:
            model = ResidualModel(arm)
            loss = model.loss(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
            with torch.no_grad():
                condition = model.encode_condition(batch)
                output = model.sample(condition, [100], 2, 42, steps=2)
            self.assertEqual(tuple(output.shape), (1, 2, 1, 24, 24))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(output.masked_select(~batch['domain'][:, None]).abs().sum().item(), 0.)

    def test_padding_and_permutation(self):
        batch = example(1)
        model = ResidualModel('det').eval()
        with torch.no_grad():
            expected = model(batch)
            changed = {k:v.clone() for k,v in batch.items()}
            changed['stations'][changed['padding']] = float('nan')
            torch.testing.assert_close(model(changed), expected, atol=2e-6, rtol=2e-5)
            order = torch.randperm(39)
            changed['stations'] = changed['stations'][:, order]
            changed['padding'] = changed['padding'][:, order]
            torch.testing.assert_close(model(changed), expected, atol=2e-6, rtol=2e-5)
            changed['padding'][:] = True
            self.assertTrue(torch.isfinite(model(changed)).all())

    def test_edm_sample_chunk_consistency(self):
        batch = example(2)
        model = ResidualModel('edm').eval()
        with torch.no_grad():
            cond = model.encode_condition(batch)
            both = model.sample(cond, [20, 21], 2, 5, steps=18)
            separate = []
            for index in range(2):
                part = {k:(v[index:index+1] if v.shape[0]==2 else v) for k,v in batch.items()}
                separate.append(model.sample(model.encode_condition(part), [20+index], 2, 5, steps=18))
        torch.testing.assert_close(both, torch.cat(separate), atol=1e-5, rtol=2e-5)

    def test_excluded_target_nan_does_not_poison_gradients(self):
        batch = example(1)
        batch['residual'][~batch['domain']] = float('nan')
        for arm in ['edm', 'det', 'gaussian']:
            model = ResidualModel(arm)
            loss = model.loss(batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_teacher_not_a_condition(self):
        batch = example(1)
        model = ResidualModel('det').eval()
        with torch.no_grad():
            before = model(batch)
            batch['residual'].fill_(1e6)
            batch['teacher_z0_mm'] = torch.full((1,1,24,24), float('nan'))
            torch.testing.assert_close(before, model(batch), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
