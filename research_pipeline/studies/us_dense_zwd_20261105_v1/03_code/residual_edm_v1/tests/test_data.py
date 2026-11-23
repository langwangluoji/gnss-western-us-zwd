import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from residual_edm_v1.data import read_rows, eligible_indices, stratified_hours, station_features, residual_scale, split_eligible, MemoryReadCache


class DataContracts(unittest.TestCase):
    def test_memory_cache_preserves_values_and_unselected_datasets_stay_lazy(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x.h5'
            with h5py.File(path,'w') as f:
                f.create_dataset('input',data=np.arange(24).reshape(8,3))
                f.create_dataset('labels',data=np.ones((8,3)))
            cache=MemoryReadCache(h5py.File(path,'r'),['input'])
            self.assertIsInstance(cache['input'],np.ndarray)
            self.assertIsInstance(cache['labels'],h5py.Dataset)
            np.testing.assert_array_equal(read_rows(cache['input'],[4,1,4]),np.arange(24).reshape(8,3)[[4,1,4]])
            self.assertFalse(cache['input'].flags.writeable)
            cache.close()
    def test_read_rows_preserves_duplicates_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            with h5py.File(Path(folder)/'x.h5', 'w') as f:
                ds=f.create_dataset('x',data=np.arange(24).reshape(8,3))
                np.testing.assert_array_equal(read_rows(ds,[4,1,4,0]),np.arange(24).reshape(8,3)[[4,1,4,0]])

    def test_history_never_crosses_year_and_anchor(self):
        years=np.repeat([2000,2001],30); hours=np.arange(60)
        ids=eligible_indices(years,hours,np.ones(60,bool),0)
        np.testing.assert_array_equal(ids,np.r_[np.arange(19,30),np.arange(49,60)])

    def test_screen_stratification_covers_all_hours(self):
        hours=np.arange(262968,262968+24*31)  # January 2000 nominal hours
        ids=stratified_hours(hours,np.arange(len(hours)))
        self.assertEqual(len(ids),48)
        self.assertEqual(len(np.unique(hours[ids]%24)),24)

    def test_partial_station_is_excluded_without_future_fill(self):
        values=np.ones((2,3,5),np.float32); values[0,1,2]=np.nan
        result,padding=station_features(values,np.zeros(5),np.ones(5),np.zeros((3,3)),np.ones((2,4)))
        self.assertEqual(result.shape,(2,3,17))
        self.assertTrue(padding[0,1]); self.assertTrue(np.all(result[0,1]==0))
        self.assertTrue(np.all(result[1,:,5:10]==1))

    def test_rms_preserves_nonzero_bias(self):
        x=np.array([2.,2.,2.]); s=residual_scale(float((x*x).sum()),len(x))
        self.assertEqual(s,4.); self.assertEqual(np.mean(x/s),.5)
        with self.assertRaises(ValueError): residual_scale(0,3)

    def test_evaluation_hours_do_not_use_label_eligibility(self):
        years=np.repeat([2018,2019,2020],30); hours=np.arange(90)
        usable=np.ones(90,bool)
        before=split_eligible(years,hours,usable,0)
        usable[30:]=False
        after=split_eligible(years,hours,usable,0)
        np.testing.assert_array_equal(before[0],after[0])
        np.testing.assert_array_equal(before[1],after[1])


if __name__=='__main__': unittest.main()
