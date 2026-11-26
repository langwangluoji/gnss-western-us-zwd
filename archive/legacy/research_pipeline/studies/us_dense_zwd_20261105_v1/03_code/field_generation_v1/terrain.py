"""Acquire official EGM96 SRTM tiles and sample registered global geometry."""
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import contains_xy
from acquire_srtm import download,BASE_URL
from field_generation_v1.prepare import DATA,ROOT,STUDY,PIPE,CRS
from residual_edm_v1.data import write_json,sha256

def grid_geometry():
    forward=Transformer.from_crs(4326,CRS,always_xy=True)
    inverse=Transformer.from_crs(CRS,4326,always_xy=True)
    lon,lat=np.meshgrid(np.linspace(-125,-110,61),np.linspace(25,50,101))
    x,y=forward.transform(lon,lat)
    # Extra half patch supplies complete centers and interpolation at region edges.
    ox=np.floor(x.min()/300000)*300000-300000
    oy=np.floor(y.min()/300000)*300000-300000
    nx=int(np.ceil((x.max()-ox)/300000)+1)*12
    ny=int(np.ceil((y.max()-oy)/300000)+1)*12
    xx,yy=np.meshgrid(ox+(np.arange(nx)+.5)*25000,oy+(np.arange(ny)+.5)*25000)
    glon,glat=inverse.transform(xx,yy)
    features=json.loads((PIPE/'studies/us_dense_zwd_20261105_plan/evidence/countries.geojson').read_text(encoding='utf-8'))['features']
    landgeom=unary_union([shape(f['geometry']) for f in features])
    land=contains_xy(landgeom,glon,glat)
    region=land&(glon>=-125)&(glon<=-110)&(glat>=25)&(glat<=50)
    origins=np.array([(r,c) for r in range(0,ny-23,12) for c in range(0,nx-23,12)])
    return dict(x=xx,y=yy,lon=glon,lat=glat,land=land,region=region,origins=origins,origin=np.array([ox,oy]))

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    geo=grid_geometry()
    tiles={}
    for idx,(lon,lat,land) in enumerate(zip(geo['lon'].ravel(),geo['lat'].ravel(),geo['land'].ravel())):
        if land: tiles.setdefault((int(np.floor((lon+180)/5))+1,int(np.floor((60-lat)/5))+1),[]).append(idx)
    dest=STUDY/'02_data/raw/srtm';dest.mkdir(parents=True,exist_ok=True)
    def acquire(key):
        col,row=key;path=dest/f'srtm_{col:02d}_{row:02d}.zip'
        download(f'{BASE_URL}/{path.name}',path)
        return key,path
    height=np.zeros(geo['lon'].size,dtype='f4');records=[];coastal_samples=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(acquire,key) for key in tiles]):
            key,path=future.result();ids=np.asarray(tiles[key])
            with zipfile.ZipFile(path) as z:
                member=next(n for n in z.namelist() if n.lower().endswith('.tif'))
            with rasterio.open(f'zip://{path}!{member}') as src:
                coords=np.c_[geo['lon'].ravel()[ids],geo['lat'].ravel()[ids]]
                values=np.array([v[0] for v in src.sample(coords)],dtype='f4')
                bad=(values==src.nodata)|~np.isfinite(values)
                for k in np.flatnonzero(bad):
                    row,col=src.index(*coords[k]);radius=150
                    window=rasterio.windows.Window(col-radius,row-radius,2*radius+1,2*radius+1)
                    a=src.read(1,window=window,boundless=True,fill_value=src.nodata)
                    yy,xx=np.where((a!=src.nodata)&np.isfinite(a))
                    if not len(xx):raise ValueError(f'No valid DEM inside25km cell: {coords[k]}')
                    distance=np.hypot(xx-radius,yy-radius);j=distance.argmin()
                    values[k]=a[yy,xx].mean(dtype='f8')
                    coastal_samples.append(dict(lon=float(coords[k,0]),lat=float(coords[k,1]),nearest_valid_pixels=float(distance[j]),height=float(values[k]),tile=path.name,method='valid terrain mean in approximately25km coastal grid footprint'))
                height[ids]=values
            records.append(dict(tile=path.name,sha256=sha256(path),grid_points=len(ids)))
            print(f'TERRAIN {len(records)}/{len(tiles)} {path.name}',flush=True)
    geo['height']=height.reshape(geo['lon'].shape)
    np.savez_compressed(DATA/'geometry.npz',**geo)
    write_json(DATA/'terrain.json',dict(tiles=records,datum='EGM96',ocean_background_height=0,coastal_nearest_valid_samples=coastal_samples,geometry_sha256=sha256(DATA/'geometry.npz')))

if __name__=='__main__':main()
