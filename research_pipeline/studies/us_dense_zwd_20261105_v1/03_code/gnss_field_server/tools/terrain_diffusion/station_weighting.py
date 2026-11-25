"""Fixed geometry-derived distance balancing; evaluation populations are unchanged."""
import torch

BIN_WEIGHTS = (.278, 2.297, 1.922, .879)


@torch.no_grad()
def distance_weights(query_xy, condition_xy, current_valid):
    distance = torch.cdist(query_xy.float(), condition_xy.float(), compute_mode='donot_use_mm_for_euclid_dist')
    near = distance[None].masked_fill(~current_valid[:, None], float('inf')).min(-1).values
    bins = torch.bucketize(near, near.new_tensor([50., 100., 200.]), right=True)
    return near.new_tensor(BIN_WEIGHTS)[bins]


def weighted_loss(prediction, target, mask, weights):
    error = prediction-target
    loss = (error[mask].square()*weights[mask]).sum()/weights[mask].sum()
    pair = mask[1:] & mask[:-1]
    if pair.any():
        pair_weight = (weights[1:]+weights[:-1])/2
        loss = loss+.1*(error.diff(dim=0)[pair].square()*pair_weight[pair]).sum()/pair_weight[pair].sum()
    return loss
