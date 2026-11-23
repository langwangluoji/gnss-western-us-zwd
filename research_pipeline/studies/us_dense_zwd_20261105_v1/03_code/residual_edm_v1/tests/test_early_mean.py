import unittest
import numpy as np
import torch
from residual_edm_v1.early_mean import early_indices, validate_early_checkpoint, paired_indices


class EarlyMeanTests(unittest.TestCase):
    def test_indices_exclude_late_year_and_cross_year_histories(self):
        years = np.repeat([2014, 2015], 30)
        hours = np.arange(60)
        usable = np.ones(60, dtype=bool)
        usable[23] = False
        ids = early_indices(years, hours, usable, anchor=0)
        np.testing.assert_array_equal(ids, np.array([19,20,21,22,24,25,26,27,28,29]))
        np.testing.assert_array_equal(paired_indices(ids[:2]), [19,18,20,19])

    def test_checkpoint_rejects_late_normalization_or_incomplete_passes(self):
        state = dict(train_years=[2000,2014], normalization_years=[2000,2014],
                     completed_passes=4, seed=17,
                     config=dict(generator='query_film', critic='none', latent=False),
                     model={'x': torch.ones(1)}, mean=torch.ones(1,1,15,1,1),
                     std=torch.ones(1,1,15,1,1), lineage={'a':'b'})
        validate_early_checkpoint(state, {'a':'b'})
        for key,value in [('train_years',[2000,2018]), ('normalization_years',[2000,2018]),
                          ('completed_passes',3), ('seed',42), ('lineage',{'a':'c'})]:
            changed = dict(state); changed[key] = value
            with self.assertRaises(ValueError):
                validate_early_checkpoint(changed, {'a':'b'})


if __name__ == '__main__':
    unittest.main()
