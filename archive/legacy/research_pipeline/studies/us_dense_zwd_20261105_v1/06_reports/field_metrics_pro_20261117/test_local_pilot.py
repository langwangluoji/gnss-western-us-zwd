import unittest
import numpy as np
from local_pilot import interpolate,query,lowpass,select_rapid_events,pair_metrics

class NumericalContracts(unittest.TestCase):
    def test_pair_members_share_the_same_heldout_model(self):
        import pandas as pd
        frame=pd.DataFrame(dict(fold=[0,1],time_index=[1,1],station=[0,1],prediction_mm=[1.,4.],target_mm=[0.,0.]))
        score,count=pair_metrics(frame,np.array([[0,0],[10000,0]]),np.zeros(2))
        self.assertEqual(count,0);self.assertIsNone(score)
    def test_weather_proxy_requires_actual_input_support(self):
        x=np.zeros((800,5));x[100]=100;x[300,0]=100000;x[300,1:]=np.nan
        selected=select_rapid_events(x)
        self.assertNotIn(300,selected);self.assertNotIn(303,selected)
        for t in selected:self.assertTrue(np.isfinite(x[[t,t-3]]).sum(1).min()>=4)
    def test_constant_field_and_query(self):
        xy=np.array([[0,0],[0,100],[100,0],[100,100]],float)
        grid=np.stack(np.meshgrid(np.arange(0,101,25),np.arange(0,101,25)),axis=-1)
        for method in ['IDW','kriging']:
            field=interpolate(xy,np.full(4,123.),grid.reshape(-1,2),100,.05,method).reshape(5,5)
            np.testing.assert_allclose(field,123,atol=1e-9)
            np.testing.assert_allclose(query(field,np.array([[1.2,2.3]])),123,atol=1e-9)
        np.testing.assert_allclose(lowpass(np.full((24,24),123.)),123,atol=1e-9)

    def test_coordinate_translation(self):
        xy=np.array([[0,0],[0,100],[100,0],[100,100]],float);target=np.array([[30,40],[70,80]],float)
        for method in ['IDW','kriging']:
            a=interpolate(xy,np.array([1,2,4,8.]),target,100,.05,method)
            b=interpolate(xy+10000,np.array([1,2,4,8.]),target+10000,100,.05,method)
            np.testing.assert_allclose(a,b,atol=1e-9)

if __name__=='__main__':unittest.main()
