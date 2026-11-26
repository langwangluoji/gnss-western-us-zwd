"""Velocity target control for the same innovation process and denoiser."""
import torch
from .joint_probability import InnovationDiffusion


def velocity_inverse(noisy,velocity,alpha):
    clean=alpha.sqrt()*noisy-(1-alpha).sqrt()*velocity
    noise=(1-alpha).sqrt()*noisy+alpha.sqrt()*velocity
    return clean,noise


class VelocityDiffusion(InnovationDiffusion):
    def loss(self,x,c):
        k=torch.randint(1,101,(len(x),),device=x.device);eps=torch.randn_like(x)
        a=self.alpha[k,None];noisy=a.sqrt()*x+(1-a).sqrt()*eps
        target=a.sqrt()*eps-(1-a).sqrt()*x
        return (self(noisy,c,k/100)-target).square().mean()

    @torch.no_grad()
    def sample(self,c,noise,steps=20):
        x=noise.clone();schedule=torch.linspace(100,0,steps+1,device=x.device).round().long()
        for k,j in zip(schedule[:-1],schedule[1:]):
            v=self(x,c,torch.full((len(x),),float(k)/100,device=x.device))
            clean,eps=velocity_inverse(x,v,self.alpha[k])
            a=self.alpha[j];x=a.sqrt()*clean+(1-a).sqrt()*eps
        return x
