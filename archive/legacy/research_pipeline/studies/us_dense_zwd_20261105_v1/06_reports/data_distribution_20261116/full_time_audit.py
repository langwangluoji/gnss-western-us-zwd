"""All raw records, timestamp coverage only; no scientific-value QC implied."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import gzip,re,json,time,calendar
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent/'full_raw_time'
S=OUT.parents[2]
SOURCE=S.parents[1]/'data/pwv_data'
BASE=int(np.datetime64('2000-01-01','h').astype(int))
END=int(np.datetime64('2022-01-01','h').astype(int))
BASES=np.array([int(np.datetime64(f'{y}-01-01','h').astype(int)) for y in range(2000,2023)])
PAT=re.compile(rb'^\s*([A-Za-z0-9_]{4})\s+(\d{2}:\d{3}:\d{5})\s+',re.M)

def scan(path):
    try:
        stat=path.stat()
        with gzip.open(path,'rb') as f:text=f.read()
        section=text.split(b'+TROP/SOLUTION',1)[1].split(b'-TROP/SOLUTION',1)[0]
        rows=PAT.findall(section)
        lines=sum(bool(x.strip()) and not x.lstrip().startswith((b'*',b'%',b'-')) for x in section.splitlines())
        if not rows or len(rows)!=lines:raise ValueError(f'Parsed {len(rows)} epochs from {lines} data lines')
        name=path.name.split('.')[0].upper()
        if any(s.decode().upper()!=name for s,e in rows):raise ValueError('Station mismatch')
        e=np.array([list(t) for _,t in rows],dtype=np.uint8)-48
        yy=e[:,0]*10+e[:,1]; day=e[:,3].astype(int)*100+e[:,4]*10+e[:,5]
        sec=e[:,7].astype(int)*10000+e[:,8].astype(int)*1000+e[:,9].astype(int)*100+e[:,10]*10+e[:,11]
        if np.any(yy>21):raise ValueError('Year outside expected 2000-2021')
        leap=np.array([366 if calendar.isleap(2000+int(y)) else 365 for y in range(22)])
        if np.any((day<1)|(day>leap[yy])|(sec>=86400)):raise ValueError('Invalid epoch')
        ts=(BASES[yy]+(day-1)*24)*3600+sec
        if np.any(sec%3600):raise ValueError('Off-hour record: requires separate audit')
        h=ts//3600; unique=np.unique(h); mask=np.zeros(END-BASE,bool);mask[unique-BASE]=True
        after=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed')
        gaps=np.diff(unique)-1
        row=dict(station=name,first=str(np.datetime64(int(unique[0]),'h')),last=str(np.datetime64(int(unique[-1]),'h')),records=len(h),unique_hours=len(unique),duplicates=len(h)-len(unique),out_of_order=int((np.diff(h)<0).sum()),internal_missing_hours=int(unique[-1]-unique[0]+1-len(unique)),max_internal_gap_hours=int(gaps.max()) if len(gaps) else 0,crc_ok=True)
        annual=[dict(station=name,year=y,record_hours=int(mask[BASES[y-2000]-BASE:BASES[y-1999]-BASE].sum()),calendar_hours=int(BASES[y-1999]-BASES[y-2000])) for y in range(2000,2022)]
        return row,annual,np.packbits(mask)
    except Exception as exc:return dict(station=path.stem,error=str(exc)),[],None

def main():
    OUT.mkdir(exist_ok=True)
    files=sorted(SOURCE.glob('*.pwv.gz'));assert len(files)>5000
    rows=[];annual=[];counts=np.zeros(END-BASE,np.int32);start=time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,(row,yrs,bits) in enumerate(pool.map(scan,files,chunksize=4),1):
            rows.append(row);annual.extend(yrs)
            if bits is not None:counts+=np.unpackbits(bits,count=END-BASE)
            if i%100==0 or i==len(files):
                progress=dict(done=i,total=len(files),failed=sum('error' in x for x in rows),seconds=round(time.time()-start,1))
                (OUT/'progress.json').write_text(json.dumps(progress),encoding='utf8');print(progress,flush=True)
    st=pd.DataFrame(rows);a=pd.DataFrame(annual)
    st.to_csv(OUT/'station_actual_time.csv',index=False);a.to_csv(OUT/'station_year_record_coverage.csv',index=False)
    dates=pd.date_range('2000-01-01','2021-12-31 23:00',freq='h')
    pd.DataFrame({'time':dates,'stations_with_record':counts}).to_csv(OUT/'hourly_station_counts.csv',index=False)
    y=a.groupby('year').agg(active_stations=('record_hours',lambda x:int((x>0).sum())),station_hours=('record_hours','sum'),full_year_stations=('record_hours',lambda x:0))
    for year in y.index:y.loc[year,'full_year_stations']=int((a[a.year==year].record_hours==a[a.year==year].calendar_hours).sum())
    y.to_csv(OUT/'annual_summary.csv')
    good=st[st.crc_ok==True];valid=np.flatnonzero(counts)
    summary=dict(files=len(files),passed=len(good),failures=[x for x in rows if 'error'in x],first=str(dates[valid[0]]),last=str(dates[valid[-1]]),total_records=int(good.records.sum()),unique_station_hours=int(good.unique_hours.sum()),global_internal_empty_hours=int((counts[valid[0]:valid[-1]+1]==0).sum()),stations_covering_2016_2020_endpoints=int(((pd.to_datetime(good['first'])<=pd.Timestamp('2016-01-01'))&(pd.to_datetime(good['last'])>=pd.Timestamp('2020-12-31 23:00'))).sum()),seconds=time.time()-start,scope='Actual raw timestamp presence and gzip CRC; no numeric QC or UTC shift')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(summary);print(y.to_string())

if __name__=='__main__':main()
