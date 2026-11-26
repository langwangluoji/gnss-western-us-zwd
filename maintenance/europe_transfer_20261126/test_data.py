import numpy as np
import pytest

from . import data


def test_missing_six_hour_epoch_is_not_compressed():
    hours=np.array([0,6,18]); stations=np.array([0,0,0])
    a=data.dense_values(hours,stations,np.array([[10,280,900],[12,281,901],[16,282,902]]),1)
    assert a.shape==(4,1,3)
    assert np.isnan(a[2]).all()
    assert a[3,0,0]==16


def test_duplicate_station_epoch_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        data.dense_values(np.array([0,0]),np.array([0,0]),np.ones((2,3)),1)


def test_pressure_uses_dry_air_and_kelvin():
    p=data.dry_pressure(np.array([1000,1000]),np.array([280,280]),np.array([0,1000]))
    assert p[0]==1000
    assert p[1]==pytest.approx(885.129,abs=.01)


def test_station_split_is_metadata_only_disjoint_reproducible():
    xy=np.random.default_rng(1).normal(size=(400,2))*1000
    roles=data.station_roles(xy)
    assert {r:int((roles==r).sum()) for r in np.unique(roles)}=={'train':280,'dev':60,'confirm':60}
    assert np.array_equal(roles,data.station_roles(xy))


def test_colocated_alias_stations_never_cross_split():
    xy=np.c_[np.arange(400)*10.,np.zeros(400)]
    xy[1]=xy[0]+[.2,0]
    roles,groups=data.site_roles(xy,np.ones(400,dtype=bool))
    assert groups[0]==groups[1]
    assert roles[0]==roles[1]
