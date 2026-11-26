import h5py
import numpy as np
from . import legacy


def test_legacy_training_copy_contains_only_registered_training_stations(tmp_path):
    source=tmp_path/'source.h5';dest=tmp_path/'filtered.h5'
    with h5py.File(source,'w') as f:
        f['station_phys']=np.array(['AAAA','BBBB','AAAA','CCCC'],dtype=h5py.string_dtype())
        f['zwd']=np.array([.1,.2,.3,.4]);f.attrs['columns']=['station_phys','zwd']
    legacy.filter_h5(source,dest,['AAAA'])
    with h5py.File(dest) as f:
        assert f['station_phys'].asstr()[:].tolist()==['AAAA','AAAA']
        np.testing.assert_allclose(f['zwd'][:],[.1,.3])
    with h5py.File(source) as f:assert len(f['zwd'])==4
