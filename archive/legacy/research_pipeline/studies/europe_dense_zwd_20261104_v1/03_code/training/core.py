"""Sparse loss, split guards and train-only moments for prepared tile batches."""
import torch

def check_split(target_groups, reference_groups, heldout_groups):
    t,r,h=map(set,(target_groups,reference_groups,heldout_groups))
    if t & r or r & h: raise ValueError('reference physical group leakage')

def fit_normalizer(train_conditions, split):
    if split != 'train': raise ValueError('normalization must use training tiles only')
    if not torch.isfinite(train_conditions).all(): raise ValueError('nonfinite input requires explicit mask/fill')
    mean=train_conditions.mean((0,1,3,4),keepdim=True)
    std=train_conditions.std((0,1,3,4),keepdim=True,unbiased=False).clamp_min(1e-6)
    return mean,std

def sparse_loss(prediction, label, valid):
    valid=valid.bool() & torch.isfinite(label)
    if not valid.any(): raise ValueError('no valid real labels')
    return ((prediction[valid]-label[valid])**2).mean()

def critic_loss(real_score,fake_score):
    return torch.relu(1-real_score).mean()+torch.relu(1+fake_score).mean()

def macro_rmse(errors, groups):
    import numpy as np
    errors=np.asarray(errors); groups=np.asarray(groups)
    if len(errors)==0 or not np.isfinite(errors).all(): raise ValueError('invalid scores')
    return float(np.mean([np.sqrt(np.mean(errors[groups==g]**2)) for g in np.unique(groups)]))


MASKED_CHANNELS=(0,1,2,3,4,6)

def fit_normalizer_stream(batches):
    """Training-only population moments; missing weather never affects statistics."""
    count=total=square_total=None
    for batch in batches:
        values=batch['conditions'].double()
        if values.ndim!=5 or values.shape[2]!=15 or not torch.isfinite(values).all():
            raise ValueError('expected finite masked 15-channel conditions')
        valid=torch.ones_like(values,dtype=torch.bool)
        valid[:,:,list(MASKED_CHANNELS)]=(values[:,:,5:6]>.5).expand(-1,-1,len(MASKED_CHANNELS),-1,-1)
        clean=torch.where(valid,values,torch.zeros_like(values))
        sums=clean.sum((0,1,3,4),keepdim=True)
        squares=clean.square().sum((0,1,3,4),keepdim=True)
        counts=valid.sum((0,1,3,4),keepdim=True).double()
        total=sums if total is None else total+sums
        square_total=squares if square_total is None else square_total+squares
        count=counts if count is None else count+counts
    if count is None or (count==0).any(): raise ValueError('no valid training values for at least one channel')
    mean=total/count
    std=(square_total/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    return mean.float(),std.float()

def normalize_conditions(raw,mean,std):
    """Normalize channels and set unavailable weather features to zero."""
    normalized=(raw-mean)/std
    normalized=normalized.clone()
    masked=normalized[:,:,list(MASKED_CHANNELS)]
    normalized[:,:,list(MASKED_CHANNELS)]=torch.where((raw[:,:,5:6]>.5).expand_as(masked),masked,torch.zeros_like(masked))
    return normalized

def valid_interpolation_masks(coarse_mask):
    """Mask fine pixels only when every contributing coarse source is valid."""
    import torch.nn.functional as functional
    fine=functional.interpolate(coarse_mask.float(),scale_factor=4,mode='bilinear',align_corners=False)>=1-1e-6
    coarse=functional.avg_pool2d(fine.float(),4,4)>=1-1e-6
    return fine,coarse

def fine_phase_penalty(fine,complete_coarse_mask):
    """Squared common-H0 phase means from identical complete 4x4 blocks, /(20 mm)^2."""
    b,c,h,w=fine.shape
    if c!=1 or h%4 or w%4 or complete_coarse_mask.shape!=(b,1,h//4,w//4): raise ValueError('fine/complete block shapes differ')
    blocks=fine.reshape(b,1,h//4,4,w//4,4).permute(0,1,2,4,3,5)
    mask=complete_coarse_mask.unsqueeze(-1).unsqueeze(-1)
    count=mask.sum()
    if count==0: raise ValueError('no complete fine blocks')
    phase_means=(blocks*mask).sum((0,1,2,3))/count
    return phase_means.square().mean()/400.

def masked_fine_gradient_penalty(fine,valid):
    """Squared 1 mm/km scaled adjacent gradients, including across block edges."""
    dx=(fine[:,:,:,1:]-fine[:,:,:,:-1])/5.; dy=(fine[:,:,1:,:]-fine[:,:,:-1,:])/5.
    vx=valid[:,:,:,1:]&valid[:,:,:,:-1]; vy=valid[:,:,1:,:]&valid[:,:,:-1,:]
    return (dx.square()[vx].mean() if vx.any() else fine.new_zeros(()))+(dy.square()[vy].mean() if vy.any() else fine.new_zeros(()))
