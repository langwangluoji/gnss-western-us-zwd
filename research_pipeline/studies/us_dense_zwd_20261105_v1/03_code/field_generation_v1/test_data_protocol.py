"""Audit the actual frozen station manifest against the spatial holdout protocol."""
import unittest
import numpy as np
import pandas as pd
from field_generation_v1.prepare import DATA

class DataProtocolTests(unittest.TestCase):
    def test_dense_groups_are_indivisible(self):
        frame=pd.read_csv(DATA/'station_roles.csv')
        rest=frame[frame.role!='input'].copy()
        rest['gx']=np.floor(rest.x/50000);rest['gy']=np.floor(rest.y/50000)
        violations=[]
        for key,group in rest.groupby(['gx','gy']):
            if len(group)>=4 and group.role.nunique()!=1:violations.append((key,group.station.tolist()))
        self.assertEqual(violations,[],f'{len(violations)} dense groups cross role boundaries')

    def test_identity_coverage_and_input_lattice(self):
        frame=pd.read_csv(DATA/'station_roles.csv')
        self.assertEqual(frame.physical_group.nunique(),len(frame))
        self.assertEqual(frame.station.nunique(),len(frame))
        self.assertTrue((frame.minimum_train_coverage>=.9).all())
        inputs=frame[frame.role=='input']
        self.assertEqual(len(inputs[['cellx','celly']].drop_duplicates()),len(inputs))

if __name__=='__main__':unittest.main()
