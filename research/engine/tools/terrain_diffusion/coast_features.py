"""Natural Earth 1:50m coastline; projected segment distance, kilometres."""
from pathlib import Path
import hashlib
import json
import urllib.request

import numpy as np
from pyproj import Transformer

CRS = '+proj=lcc +lat_0=37.5 +lon_0=-117.5 +lat_1=30 +lat_2=45 +datum=WGS84 +units=m'
URL = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson'


def distance_to_coast(points, lines):
    segments = [(line[:-1], line[1:]) for line in lines if len(line) > 1]
    a = np.concatenate([x[0] for x in segments])
    b = np.concatenate([x[1] for x in segments])
    d = b - a
    length2 = np.maximum((d*d).sum(1), 1e-12)
    result = []
    for p in np.asarray(points):
        t = np.clip(((p-a)*d).sum(1)/length2, 0, 1)
        result.append(np.sqrt(((p-a-t[:, None]*d)**2).sum(1)).min())
    return np.asarray(result, dtype='float32')


def load_coast(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'ne_50m_coastline.geojson'
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=90) as response:
            path.write_bytes(response.read())
    payload = json.loads(path.read_text(encoding='utf-8'))
    transformer = Transformer.from_crs(4326, CRS, always_xy=True)
    lines = []
    for feature in payload['features']:
        geo = feature['geometry']
        coords = [geo['coordinates']] if geo['type'] == 'LineString' else geo['coordinates']
        for line in coords:
            a = np.asarray(line)
            # Retain entire lines touching a wide buffer; never clip segments
            # at the study boundary and create an artificial coast.
            hit = (a[:, 0] > -150) & (a[:, 0] < -90) & (a[:, 1] > 10) & (a[:, 1] < 65)
            if hit.any():
                x, y = transformer.transform(a[:, 0], a[:, 1])
                lines.append(np.c_[x, y]/1000)
    receipt = dict(url=URL, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   license='Natural Earth public domain', scale='1:50m',
                   distance='minimum projected LCC segment distance, km; not geodesic',
                   crs=CRS, segments=sum(len(x)-1 for x in lines))
    return lines, receipt
