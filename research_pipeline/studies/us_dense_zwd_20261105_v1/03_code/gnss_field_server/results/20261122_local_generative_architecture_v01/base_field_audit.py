"""Descriptive input-supported roughness audit; not a dense-truth accuracy test."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnss_field.artifacts import read_json,write_json
from gnss_field.data import PackedStore,ContextBuilder
from grid_pair_data import local_runtime
from pilot_data import ROOT,SOURCE,PROJECT


def main():
    case=np.load(ROOT/'fold_0/G0/field_case.npz');a=np.load(SOURCE/'data.npz');hour=int(case['hour'][0]);ix=int(np.searchsorted(a['hours'],hour))
    m0=np.load(SOURCE/'cache/M0.npy',mmap_mode='r')[ix].copy();features=np.load(ROOT/'cache/conditions69.npy',mmap_mode='r')[ix]
    settings=read_json(PROJECT/'results/20261120_local_context_three_arm_v01/dataset/settings.json')
    idw=features[59]*settings['grid_scale'];mixture=m0+settings['baseline_weight']*(idw-m0)
    store=PackedStore(local_runtime()['data_root'],purpose='context');support=ContextBuilder(store).support(hour)[0]&store.grid['region']
    fig,axes=plt.subplots(1,4,figsize=(13,6));records={}
    for ax,value,name in zip(axes,[m0,idw,mixture,case['mean_Z0'].squeeze()],['Frozen M0','A-IDW condition','Exact frozen mixture','K8 ensemble mean']):
        im=ax.imshow(np.where(support,value,np.nan),origin='lower',vmin=0,vmax=150);ax.set_title(name);ax.set_xticks([]);ax.set_yticks([])
        dxx=value[:,2:]-2*value[:,1:-1]+value[:,:-2];dyy=value[2:]-2*value[1:-1]+value[:-2]
        sx=support[:,2:]&support[:,1:-1]&support[:,:-2];sy=support[2:]&support[1:-1]&support[:-2]
        records[name]=dict(x_second_difference_rms_mm=float(np.sqrt(np.mean(dxx[sx]**2))),y_second_difference_rms_mm=float(np.sqrt(np.mean(dyy[sy]**2))))
    fig.subplots_adjust(right=.92);fig.colorbar(im,cax=fig.add_axes([.94,.15,.015,.7]),label='Z0 mm')
    fig.suptitle('Same input-supported land; descriptive output audit, no dense HR truth');fig.savefig(ROOT/'base_field_audit.png',dpi=160);plt.close(fig)
    records['settings']=dict(baseline_weight=settings['baseline_weight'],hour=hour,supported_land_nodes=int(support.sum()),scope='descriptive roughness, not accuracy or proof of seams')
    write_json(ROOT/'base_field_audit.json',records)


if __name__=='__main__':main()
