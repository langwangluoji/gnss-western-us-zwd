"""CPU-only registered result figures and calendar-case field exports."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse
import io

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from gnss_field.artifacts import atomic_bytes,write_json
from gnss_field.generative.campaign import Campaign,load_config,build_data
from gnss_field.generative.training import load_stage_model
from gnss_field.generative.evaluation import _latent_fields


def spectrum(field,land):
    window=np.outer(np.hanning(field.shape[0]),np.hanning(field.shape[1]))
    centered=np.where(land,field-float(field[land].mean()),0.)*window
    power=np.abs(np.fft.rfft2(centered))**2/max(float((window*land).sum())**2,1.)
    ky=np.fft.fftfreq(field.shape[0],d=25.)[:,None]
    kx=np.fft.rfftfreq(field.shape[1],d=25.)[None]
    radial=np.sqrt(ky*ky+kx*kx)
    edges=np.linspace(0,radial.max(),25)
    return [(float((lo+hi)/2),float(power[(radial>=lo)&(radial<hi)].mean()))
             for lo,hi in zip(edges[1:-1],edges[2:]) if ((radial>=lo)&(radial<hi)).any()]


def export_fields(campaign):
    config=campaign.config;config['runtime']['device']='cpu'
    data=build_data(config,purpose='development')
    reports=campaign.manifest['stages']['final_evaluation']['reports']
    root=campaign.root/'evaluation/calendar_cases';root.mkdir(parents=True,exist_ok=True)
    cases=['2019-01-15T12','2019-04-15T12','2019-07-15T12','2019-10-15T12']
    outputs=[]
    for family in ('vae','gan','diffusion'):
        stage='complete_'+family
        path=reports[stage]['products']['joint']['checkpoint']
        model,_=load_stage_model(path,'cpu')
        for date in cases:
            hour=int(np.datetime64(date,'h').astype('i8'))
            # Inference reads conditions only: C labels do not reach this path.
            batch=data.condition_batch([hour],device='cpu')
            fields=_latent_fields(model,batch,16,2026112242,18,4).numpy()[0,:,0]
            mean=fields.mean(0);std=fields.std(0,ddof=1)
            background=batch['background'].numpy()[0,0]
            land=data.land;key=f'{family}_{date.replace(":","")}'
            buffer=io.BytesIO();np.savez_compressed(buffer,hour=hour,members=fields,mean=mean,std=std,
                background=background,land=land,xy=data.grid_geometry[...,:2],height=data.grid_geometry[...,2])
            atomic_bytes(root/(key+'.npz'),buffer.getvalue())
            fig=Figure(figsize=(13,4),layout='constrained');FigureCanvasAgg(fig)
            views=(background,mean,mean-background,std)
            for index,(value,title) in enumerate(zip(views,('Fusion background','Generated mean','Mean correction','Field spread'))):
                axis=fig.add_subplot(1,4,index+1)
                im=axis.imshow(np.where(land,value,np.nan),origin='lower',cmap='RdBu_r' if index==2 else 'viridis')
                axis.set_title(title);axis.set_xticks([]);axis.set_yticks([]);fig.colorbar(im,ax=axis,shrink=.6,label='Common-height ZWD (mm)')
            fig.suptitle(f'Conditional {family.upper()} | {date} | 16 prior members | 25 km output spacing')
            field_path=root/(key+'_field.png');fig.savefig(field_path,dpi=170);fig.clear()
            fig=Figure(figsize=(6,4),layout='constrained');FigureCanvasAgg(fig);axis=fig.add_subplot()
            for field,label in ((background,'Fusion background'),(mean,'Generated mean'),(fields[0],'One generated member')):
                values=np.asarray(spectrum(field,land));axis.loglog(values[:,0],values[:,1],label=label)
            axis.set(xlabel='Spatial frequency (cycles/km)',ylabel='Windowed spectral power (mm²)',
                     title='Same land mask and Hann window; structural diagnostic')
            axis.legend();psd=root/(key+'_spectrum.png');fig.savefig(psd,dpi=170);fig.clear()
            outputs.append(dict(family=family,hour=hour,field_png=str(field_path),spectrum_png=str(psd),members_npz=str(root/(key+'.npz'))))
        del model
    result=dict(calendar_cases=cases,selection='registered dates; not selected by model error',
        field_members=16,field_spread_excludes_station_discrepancy=True,outputs=outputs,
        PSD_scope='same masked/windowed computational grid; no independent dense truth')
    write_json(root/'manifest.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    from gnss_field.generative.report import export_campaign_figures
    campaign=Campaign(load_config(args.config))
    if campaign.manifest['stages'].get('final_evaluation',{}).get('status')!='complete':
        raise ValueError('Complete locked C2019 evaluation is required before final report')
    try:
        report=dict(status='complete',station_figures=export_campaign_figures(campaign),calendar_fields=export_fields(campaign),
                    D_labels_read=False,CPU_only=True)
        write_json(campaign.root/'results_export.json',report)
        campaign.manifest['results_export']=dict(status='complete',path=str(campaign.root/'results_export.json'));campaign.save()
        print('RESULTS_EXPORTED',str(campaign.root/'results_export.json'),flush=True)
    except Exception as exc:
        campaign.manifest['results_export']=dict(status='error',reason=str(exc));campaign.save();raise


if __name__=='__main__':main()
