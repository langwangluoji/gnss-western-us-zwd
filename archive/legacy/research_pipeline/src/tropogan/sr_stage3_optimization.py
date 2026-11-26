"""Measured, one-sided adversarial gradient guard for sparse GNSS GANs.

This is a local Euclidean gradient rule, not a guarantee about Adam updates,
validation error, measurement noise, or scientific benefit. The supervised
objective and all observations remain unchanged by the projection itself.
"""
from __future__ import annotations

import math
import torch


def quadratic_matched_pseudo_huber(error, delta=2.):
    """Dimensionless loss, equal to error**2 near zero; no sample removal.

    For errors expressed as (prediction_mm-truth_mm)/10, delta=2 means 20 mm.
    The algebraic form avoids cancellation for small errors.
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    return 2*error.square()/(torch.sqrt(1+(error/delta).square())+1)


def _dots(left, right):
    return sum((a.double()*b.double()).sum() for a, b in zip(left, right))


def project_adversarial_gradients(supervised, adversarial, *, weight, ratio_cap):
    """Return guarded weighted adversarial gradient and scalar diagnostics.

    First remove a conflicting component along the full supervised gradient;
    then cap its global norm relative to the supervised gradient. For zero
    supervised norm, return zero adversarial update. All inputs are detached
    gradient tuples of identical length, shapes and device.
    """
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(ratio_cap) or ratio_cap <= 0:
        raise ValueError("Nonnegative weight and positive ratio_cap required")
    if not supervised or len(supervised) != len(adversarial):
        raise ValueError("Nonempty matching gradient tuples required")
    for s, a in zip(supervised, adversarial):
        if s.shape != a.shape or s.device != a.device or s.dtype != a.dtype:
            raise ValueError("Gradient shapes, devices and dtypes must agree")
        if s.requires_grad or a.requires_grad or not torch.isfinite(s).all() or not torch.isfinite(a).all():
            raise ValueError("Gradients must be finite and detached")
    weighted = tuple(a.double()*weight for a in adversarial)
    ss, aa, sa = _dots(supervised, supervised), _dots(adversarial, adversarial), _dots(supervised, weighted)
    s_norm, a_norm = ss.sqrt(), aa.sqrt()
    if not all(torch.isfinite(value) for value in (ss, aa, sa)):
        raise ValueError("Gradient norm overflow")
    coefficient = torch.minimum(sa, sa.new_zeros(()))/ss if float(ss) > 0. else ss.new_zeros(())
    projected = tuple(a-coefficient*s.double() for s, a in zip(supervised, weighted))
    projected_norm = _dots(projected, projected).sqrt()
    factor = ((ratio_cap*s_norm/projected_norm).clamp(max=1.) if float(projected_norm) > 0.
              else projected_norm.new_ones(()))
    if float(s_norm) == 0.:
        factor = factor*0
    guarded = tuple((a*factor).to(s.dtype) for s, a in zip(supervised, projected))
    guarded_norm = _dots(guarded, guarded).sqrt()
    guarded_dot = _dots(supervised, guarded)
    unweighted_dot = _dots(supervised, adversarial)
    sn, an = float(s_norm), float(a_norm)
    diagnostics = {
        "supervised_gradient_l2": sn,
        "adversarial_gradient_l2_unweighted": an,
        "adversarial_gradient_l2_weighted": weight*an,
        "weighted_adversarial_to_supervised_gradient_ratio": weight*an/sn if sn else None,
        "adversarial_supervised_gradient_cosine": float(unweighted_dot)/(sn*an) if sn and an else None,
        "gradient_projection_applied": bool(float(sa) < 0),
        "adversarial_gradient_l2_after_projection": float(projected_norm),
        "adversarial_gradient_cap_factor": float(factor),
        "adversarial_gradient_l2_guarded": float(guarded_norm),
        "guarded_adversarial_to_supervised_ratio": float(guarded_norm)/sn if sn else None,
        "supervised_dot_guarded_adversarial": float(guarded_dot),
        "gradient_rule": "one_sided_euclidean_projection_then_norm_cap",
    }
    return guarded, diagnostics


def assign_guarded_gradients(supervision, regularizer, adversarial, model, lambda_adv, ratio_cap):
    """Set G .grad = grad(supervision)+grad(regularizer)+guarded_adv.

    The caller zeroes G gradients first, freezes D during its G forward and
    clips the final gradient as usual. No optimizer step is performed here.
    The final scalar loss is a diagnostic: the applied projected vector need
    not be the gradient of that scalar. D parameters are never differentiated.
    """
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if not parameters or not supervision.requires_grad:
        raise ValueError("Trainable G and differentiable supervision required")
    if any(value.numel() != 1 or not torch.isfinite(value) for value in (supervision, regularizer, adversarial)):
        raise ValueError("Finite scalar losses required")

    def gradients(value, retain):
        raw = (torch.autograd.grad(value, parameters, retain_graph=retain, allow_unused=True)
               if value.requires_grad else (None,)*len(parameters))
        return tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, raw))

    sup = gradients(supervision, True)
    reg = gradients(regularizer, True)
    adv = gradients(adversarial, False)
    guarded, diagnostics = project_adversarial_gradients(sup, adv, weight=float(lambda_adv), ratio_cap=float(ratio_cap))
    for parameter, s, r, a in zip(parameters, sup, reg, guarded):
        parameter.grad = s+r+a
    diagnostics["regularizer_gradient_l2"] = float(_dots(reg, reg).sqrt())
    return diagnostics
