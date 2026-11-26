import importlib
import importlib.util
import numpy as np


def module():
    name=__package__+'.native_innovation_audit'
    assert importlib.util.find_spec(name) is not None, 'leave-out audit module is missing'
    return importlib.import_module(name)


def test_geometric_folds_preserve_physical_groups_and_ignore_context_order():
    f=module().spatial_groups
    ids=np.array([8,2,7,3,9,1]);xy=np.array([[0.,0.],[1.,0.],[10.,0.],[11.,0.],[20.,0.],[21.,0.]])
    groups=np.array([11,11,12,12,13,13])
    a=f(ids,xy,groups,3);order=np.array([3,0,5,2,1,4]);b=f(ids[order],xy[order],groups[order],3)
    assert len(np.unique(a))==3
    np.testing.assert_array_equal(a[order],b)
    for g in np.unique(groups):assert len(np.unique(a[groups==g]))==1


def test_leave_sets_drop_station_or_entire_fold_and_preserve_sorted_ids():
    m=module();context=np.array([1,3,4,9,12]);folds=np.array([0,0,1,1,2])
    single=m.retained_context(context,np.array([4]))
    group=m.retained_context(context,context[folds==0])
    np.testing.assert_array_equal(single,[1,3,9,12]);np.testing.assert_array_equal(group,[4,9,12])
