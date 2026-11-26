"""Joint H/S arm, matching both original single-arm initial parameter draws."""
import torch
from .native_mean_fusion_v2 import NativeMeanFusionV2


class NativeMeanHS(NativeMeanFusionV2):
    def __init__(self,kind='transformer',width=64,length=72):
        initial=torch.get_rng_state()
        super().__init__('H',kind,width,length)
        # Reconstruct S from the same pre-construction CPU RNG state. Creating H
        # first must not change S's random initialization in the combined arm.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial)
            spatial=NativeMeanFusionV2('S',kind,width,length)
            self.time=spatial.time

    def new_parameter_names(self):
        return {name for name,_ in self.named_parameters() if name.startswith(('current_residual.','time.spatial.'))}
