"""NGL SINEX-TRO parsing and v2-compatible reference conversion, never native-wet mixing.

This module does not modify EPN v2, infer UTC from a filename, round observation
times, fit cross-centre corrections, or turn formal sigmas into accuracy claims.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_height_corrected_reference import (
    ecef_geodetic, orthometric_to_geopotential_approx, saastamoinen_zhd,
)
from integrate_profile_pilot import load_gtx

COMPATIBLE_SECONDS = (1800,19800,23400,41400,45000,63000,66600,84600)


def sinex_epoch(token):
    yy,dd,ss = token.split(":")
    year = int(yy)
    if year < 100: year += 2000 if year < 80 else 1900
    day, second = int(dd), int(ss)
    if day < 1 or day > (366 if pd.Timestamp(year,12,31).dayofyear == 366 else 365):
        raise ValueError(f"Invalid SINEX day: {token}")
    if not 0 <= second < 86400:
        raise ValueError(f"Nonstandard seconds-of-day require explicit handling: {token}")
    return pd.Timestamp(year,1,1)+pd.Timedelta(days=day-1,seconds=second)


@lru_cache(maxsize=4)
def geoid_loader(path):
    return load_gtx(Path(path))


def _number(token, scale=1.):
    value=float(token)
    return np.nan if value in {-999.,-9999.,99999.} else value*scale


def _parse_daily(text, source_member, keep_seconds_of_day=None):
    section=""; rows=[]; coords={}; app={}; eccentricities={}; solution_header=[]
    metadata={"source_member":source_member,"time_system_header":"UNSPECIFIED"}
    counters=Counter()
    for lineno, rawline in enumerate(text.splitlines(),1):
        line=rawline.strip()
        if not line: continue
        if line.startswith("%=TRO"):
            fields=line.split(); metadata['sinex_version']=fields[1]
            metadata['creation_epoch_raw']=fields[3]
            metadata['product_header']=line
        if line.startswith('+'): section=line[1:];continue
        if line.startswith('-'): section="";continue
        if line.startswith('*'):
            if section=='TROP/SOLUTION' and 'TROTOT' in line:
                solution_header=line[1:].split()
                metadata['solution_header']=solution_header
            continue
        fields=line.split()
        if section=='FILE/REFERENCE' and fields[0] in {'SOFTWARE','INPUT','OUTPUT'}:
            metadata[fields[0].lower()]=' '.join(fields[1:])
        elif section=='TROP/DESCRIPTION':
            if line.startswith('TIME SYSTEM'):metadata['time_system_header']=fields[-1]
            if line.startswith('SAMPLING TROP') or line.startswith('TROPO SAMPLING INTERVAL'):
                metadata['sampling_seconds']=int(fields[-1])
            if line.startswith('SOLUTION_FIELDS_1'):metadata['description_fields_1']=fields[1:]
        elif section=='SITE/ID':
            app[fields[0]]=float(fields[-1])
        elif section=='SITE/ECCENTRICITY':
            if len(fields)>=10:
                eccentricities[fields[0]]=(fields[6],*map(float,fields[7:10]))
        elif section=='TROP/STA_COORDINATES':
            xyz=list(map(float,fields[4:7]));lat,lon,height=ecef_geodetic(xyz)
            coords[fields[0]]={"x_m":xyz[0],"y_m":xyz[1],"z_m":xyz[2],"lat":lat,"lon":lon,
                "height_ellipsoidal_m":height,"frame_header":fields[7]}
        elif section=='TROP/SOLUTION':
            # NGL's descriptive block can contradict the actual table. Only the
            # verified actual header layout is accepted here.
            if len(solution_header)<5 or solution_header[2]!='TROTOT' or solution_header[3] not in {'_SIG','STDEV','STDDEV'} or solution_header[4] not in {'TRWET','TROWET'}:
                raise ValueError(f"Unrecognized NGL solution header in {source_member}: {solution_header}")
            if len(fields)!=len(solution_header):
                raise ValueError(f"Header/data length mismatch in {source_member}:{lineno}")
            counters['raw_records']+=1
            epoch=sinex_epoch(fields[1]); sec=epoch.hour*3600+epoch.minute*60+epoch.second
            if keep_seconds_of_day is not None and sec not in keep_seconds_of_day:continue
            if fields[0] not in coords:raise ValueError(f"No daily ECEF coordinate for {fields[0]}")
            row={"station":fields[0],"epoch_label":epoch,"sinex_epoch_raw":fields[1],
                "ztd_m":_number(fields[2],.001),"ztd_formal_sigma_m":_number(fields[3],.001),
                "zwd_product_m":_number(fields[4],.001),"source_member":source_member,"source_line":lineno,
                **coords[fields[0]]}
            row['zhd_product_m']=row['ztd_m']-row['zwd_product_m']
            row['height_app_m']=app.get(fields[0],np.nan)
            row['height_app_minus_ecef_m']=row['height_app_m']-row['height_ellipsoidal_m']
            ecc=eccentricities.get(fields[0],('UNSPECIFIED',np.nan,np.nan,np.nan))
            row.update(dict(zip(['eccentricity_axes','eccentricity_1_m','eccentricity_2_m','eccentricity_3_m'],ecc)))
            row['qc_eccentricity_zero']=bool(np.all(np.asarray(ecc[1:])==0))
            rows.append(row)
    if not rows and not counters['raw_records']:raise ValueError(f"No NGL solution records: {source_member}")
    for row in rows:
        for key in ['time_system_header','sinex_version','creation_epoch_raw','software','input']:
            row[key]=metadata.get(key,'UNSPECIFIED')
    metadata['rows_raw']=counters['raw_records'];metadata['rows_selected']=len(rows)
    metadata['daily_coordinates']=coords
    return rows,metadata


def parse_ngl_archive(path, geoid_path=None, keep_seconds_of_day=None):
    """Return (DataFrame,audit) from station-year ZIP, .trop.gz, or plain .trop.

    No target exclusion is implied by successful parsing. The caller must apply
    fixed network roles and alias/coordinate co-location checks before use.
    """
    path=Path(path); all_rows=[]; days=[]
    keep=None if keep_seconds_of_day is None else set(map(int,keep_seconds_of_day))
    def consume(blob,name):
        if name.lower().endswith('.gz'):blob=gzip.decompress(blob)
        rows,meta=_parse_daily(blob.decode('ascii',errors='strict'),name,keep)
        all_rows.extend(rows);days.append(meta)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith('/') and (name.lower().endswith('.trop.gz') or name.lower().endswith('.trop')):
                    consume(archive.read(name),name)
    else:consume(path.read_bytes(),path.name)
    frame=pd.DataFrame(all_rows)
    if frame.empty:raise ValueError(f"No selected records: {path}")
    duplicates=frame.duplicated(['station','epoch_label'],keep=False)
    if duplicates.any():raise ValueError(f"Duplicate station/epoch records require explicit version selection: {int(duplicates.sum())}")
    frame=frame.sort_values(['station','epoch_label']).reset_index(drop=True)
    frame['time_basis_status']=np.where(frame.time_system_header.eq('UNSPECIFIED'),'unconfirmed_raw_label','header_declared_'+frame.time_system_header)
    frame['qc_height_app_agrees']=frame.height_app_minus_ecef_m.abs().lt(2.)
    frame['qc_ztd_finite_positive']=np.isfinite(frame.ztd_m)&frame.ztd_m.gt(0)
    frame['qc_sigma_finite_nonnegative']=np.isfinite(frame.ztd_formal_sigma_m)&frame.ztd_formal_sigma_m.ge(0)
    geoid_meta=None
    if geoid_path is not None:
        separation,geoid_meta=geoid_loader(str(Path(geoid_path).resolve()))
        # Daily coordinates change slightly; compute once per unique daily XYZ.
        coordinate_rows=frame[['x_m','y_m','z_m','lat','lon','height_ellipsoidal_m']].drop_duplicates()
        coordinate_rows['geoid_undulation_m']=[separation(la,lo) for la,lo in coordinate_rows[['lat','lon']].itertuples(index=False,name=None)]
        frame=frame.merge(coordinate_rows[['x_m','y_m','z_m','geoid_undulation_m']],on=['x_m','y_m','z_m'],how='left',validate='many_to_one')
        frame['height_orthometric_m']=frame.height_ellipsoidal_m-frame.geoid_undulation_m
        frame['height_geopotential_m']=orthometric_to_geopotential_approx(frame.height_orthometric_m,frame.lat)
    frame['source_archive']=str(path.resolve())
    audit={"source_archive":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_unit_rule":"Legacy IGS SINEX TRO: TROTOT/TROWET and formal sigma in mm, converted to m; actual NGL solution header required",
        "unit_source":"https://files.igs.org/pub/data/format/sinex_tropo.txt",
        "rows_raw":sum(d['rows_raw'] for d in days),"rows_selected":len(frame),"daily_files":len(days),
        "station_ids":sorted(frame.station.unique().tolist()),"frames":frame.frame_header.value_counts().to_dict(),
        "software":frame.software.value_counts().to_dict(),"time_basis":frame.time_basis_status.value_counts().to_dict(),
        "height_app_minus_ecef_abs_max_m":float(frame.height_app_minus_ecef_m.abs().max()),
        "nonzero_or_missing_eccentricity_rows":int((~frame.qc_eccentricity_zero).sum()),
        "geoid":geoid_meta,"days":days,
        "limitations":["Raw labels are not silently tagged UTC; confirm NGL and EPN time systems before claiming physical simultaneity.",
            "Native wet/dry/PWV products are diagnostic-only and never main features or mixed reference labels.",
            "Formal sigma is conditional processing uncertainty, not measured accuracy or an EPN-NGL inter-centre bias bound.",
            "IGS20 directory name does not override the frame/software declared in each daily file.",
            "NGL README reports swapped east/north gradient columns; gradients are deliberately not exported by this parser.",
            "Nonzero/missing eccentricity needs receiver-marker/antenna delay-endpoint review; do not silently apply a guessed height offset."]}
    return frame,audit


def match_reference_epochs(frame, requests, *, requested_time_column='observation_time', time_basis_note='unconfirmed_raw_labels_only'):
    """Exact label join only. Requests contain reference station and requested raw epoch.

    To match an EPN nominal epoch, first use its audited per-row +/-30-minute
    offset. This function never rounds to 5 minutes or chooses nearest samples.
    """
    if frame.duplicated(['station','epoch_label']).any():raise ValueError('Duplicate reference epochs')
    out=requests.copy();out[requested_time_column]=pd.to_datetime(out[requested_time_column])
    out=out.merge(frame,left_on=['station',requested_time_column],right_on=['station','epoch_label'],how='left',validate='many_to_one')
    out['reference_exact_label_match']=out.epoch_label.notna()
    out['reference_time_basis_note']=time_basis_note
    out['reference_time_alignment_seconds']=(out.epoch_label-out[requested_time_column]).dt.total_seconds()
    return out


def attach_v2_zhd(frame, meteorology, *, met_time_column='epoch_label', meteorology_time_basis='unconfirmed_raw_label_alignment'):
    """Apply the frozen v2 physical operator at each *NGL station*, without fitting.

    meteorology requires station,time,sp_pa,t2m_k,wvp_hpa,orography_m. It must
    already be spatially sampled at the NGL receiver and temporally aligned to
    epoch_label using a separately documented operator. Nominal 6h values cannot
    silently be called exact-halfhour ERA5 meteorology.
    """
    required=['station',met_time_column,'sp_pa','t2m_k','wvp_hpa','orography_m']
    missing=[c for c in required if c not in meteorology]
    if missing:raise ValueError(f'Missing meteorology fields: {missing}')
    for col in ['height_orthometric_m','height_geopotential_m','lat','ztd_m']:
        if col not in frame:raise ValueError(f'Missing validated reference coordinate: {col}')
    met=meteorology[required].rename(columns={met_time_column:'epoch_label'}).copy()
    met['epoch_label']=pd.to_datetime(met.epoch_label)
    if met.duplicated(['station','epoch_label']).any():raise ValueError('Meteorology duplicate station/epoch')
    if set(['sp_pa','t2m_k','wvp_hpa','orography_m'])&set(frame.columns):raise ValueError('Frame already has meteorology; refuse ambiguous overwrite')
    out=frame.merge(met,on=['station','epoch_label'],how='left',validate='many_to_one')
    e=out.wvp_hpa*100;den=out.sp_pa-.378*e
    valid=np.isfinite(out[['sp_pa','t2m_k','wvp_hpa','orography_m','height_geopotential_m','height_orthometric_m']]).all(axis=1)&out.sp_pa.gt(0)&out.t2m_k.gt(0)&den.gt(0)&e.ge(0)&e.lt(out.sp_pa)
    q=.622*e/den;tv=out.t2m_k*(1+.61*q)
    p=out.sp_pa/100*np.exp(-9.80665*(out.height_geopotential_m-out.orography_m)/(287.05*tv))
    out['pressure_station_v2_hpa']=p.where(valid)
    out['zhd_v2_operator_m']=saastamoinen_zhd(out.pressure_station_v2_hpa,out.lat,out.height_orthometric_m)
    out['zwd_reference_v2_operator_m']=out.ztd_m-out.zhd_v2_operator_m
    out['qc_meteorology_matched_valid']=valid
    out['qc_zwd_reference_negative']=out.zwd_reference_v2_operator_m.lt(0)
    out['meteorology_time_basis']=meteorology_time_basis
    out['reference_formula']='NGL_ZTD_minus_ERA5_ZHD_using_frozen_v2_height_operator_at_NGL_station'
    return out


def audit_target_colocation(reference_meta, target_meta, radius_km=1., alias_column='original_name'):
    """Exclude reference codes/aliases or <=1km proximity to ANY target station.

    Pass every member/metadata location of every fixed train/validation/test
    target group, not only group centroids. Successful parsing does not waive this.
    """
    refs=reference_meta.reset_index(drop=True).copy();targets=target_meta.reset_index(drop=True).copy()
    if targets.empty:raise ValueError('Target metadata cannot be empty')
    a=np.deg2rad(refs[['lat','lon']].to_numpy());b=np.deg2rad(targets[['lat','lon']].to_numpy())
    h=np.sin((a[:,None,0]-b[None,:,0])/2)**2+np.cos(a[:,None,0])*np.cos(b[None,:,0])*np.sin((a[:,None,1]-b[None,:,1])/2)**2
    dist=6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
    nearest=dist.argmin(axis=1);refs['nearest_target_distance_km']=dist[np.arange(len(refs)),nearest]
    refs['nearest_target_station']=targets.station.to_numpy()[nearest]
    target_aliases=set(targets.station.astype(str).str.upper())
    if alias_column in targets:target_aliases.update(str(v).upper() for v in targets[alias_column].dropna() if str(v))
    exact=refs.station.astype(str).str.upper().isin(target_aliases)
    if alias_column in refs:exact|=refs[alias_column].fillna('').astype(str).str.upper().isin(target_aliases-{""})
    refs['same_code_or_alias_as_target']=exact
    refs['excluded_target_colocation']=exact|refs.nearest_target_distance_km.le(radius_km)
    refs['colocation_radius_km']=radius_km
    return refs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--geoid',type=Path,default=ROOT/'data/external/egm96_15.gtx')
    parser.add_argument('--compatible-times-only',action='store_true')
    args=parser.parse_args()
    frame,audit=parse_ngl_archive(args.archive,args.geoid,COMPATIBLE_SECONDS if args.compatible_times_only else None)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.audit.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(args.output,index=False);args.audit.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['rows_raw','rows_selected','daily_files','frames','time_basis','height_app_minus_ecef_abs_max_m','nonzero_or_missing_eccentricity_rows']},indent=2))


if __name__=='__main__':main()
