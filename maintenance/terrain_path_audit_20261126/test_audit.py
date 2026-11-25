import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd


def api():
    path=Path(__file__).with_name('audit.py')
    assert path.exists(), 'terrain path audit implementation is missing'
    spec=importlib.util.spec_from_file_location('terrain_path_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_dem_interpolation_uses_pixel_centers_and_marks_outside_support():
    a=api();z=np.array([[0,1,2],[10,11,12],[20,21,22]],float)
    value,clipped=a.bilinear_dem(z,np.array([[25,25],[-1,25],[62.5,62.5]]),np.zeros(2),25)
    np.testing.assert_allclose(value,[5.5,5,22]);assert clipped.tolist()==[False,True,False]


def test_path_barriers_have_meter_units_and_endpoint_sensitivity():
    a=api();z=np.tile([100,400,100],(3,1)).astype(float)
    result=a.path_features(z,np.array([12.5,37.5]),np.array([62.5,37.5]),100,100,np.zeros(2),25,31)
    assert result['barrier_m']==300 and result['above_max_station_m']==300
    assert result['dem_chord_barrier_m']==300 and not result['clipped']


def test_nearest_condition_station_uses_gnss_visibility_not_just_geometry():
    a=api();q=np.array([[0.,0.]]);c=np.array([[1.,0.],[2.,0.],[2.,0.]])
    visible=np.array([[True,True,True],[False,True,True],[False,False,False]])
    index,distance=a.nearest_visible(q,c,visible,np.array(['Z','B','A']))
    assert index[:,0].tolist()==[0,2,-1]
    np.testing.assert_allclose(distance[:2,0],[1,2]);assert np.isinf(distance[2,0])


def test_controlled_comparison_requires_common_distance_month_height_support():
    a=api();rows=[]
    for cell in [0,1]:
        for group in [0,1]:
            for station in [0,1]:rows.append(dict(cell=cell,barrier_group=group,station=station,
                month=0,n=48,signed=48*group*2,square=48*group*4))
    rows.append(dict(cell=2,barrier_group=1,station=3,month=0,n=48,signed=10000,square=1000000))
    result=a.controlled_contrast(pd.DataFrame(rows))
    assert result['supported_cells']==2
    assert result['bias_high_minus_low_mm']==2
    assert result['rmse_high_minus_low_mm']==2
