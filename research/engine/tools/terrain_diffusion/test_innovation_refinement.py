import torch


def test_error_field_height_transport_and_query_independence():
    from . import innovation_refinement as m
    cxy=torch.tensor([[0.,0.],[70.,30.],[120.,80.]])
    ch=torch.tensor([0.,700.,1700.]);qh=torch.tensor([100.,900.]);qxy=torch.tensor([[10.,0.],[80.,60.]])
    error=2*torch.exp(-ch/3000)[None].expand(2,-1).clone();valid=torch.ones_like(error,dtype=torch.bool)
    error[1,0]=torch.nan;valid[1,0]=False
    fields=m.innovation_fields(error,valid,qxy,cxy,qh,ch)
    expected=2*torch.exp(-qh/3000)[None,:,None].expand_as(fields)
    torch.testing.assert_close(fields,expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(m.innovation_fields(error,valid,qxy[:1],cxy,qh[:1],ch),fields[:,:1])
    assert torch.equal(m.innovation_fields(error,torch.zeros_like(valid),qxy,cxy,qh,ch),torch.zeros_like(fields))


def test_lagged_innovation_features_are_causal_and_preserve_zero_fallback():
    from . import innovation_refinement as m
    fields=torch.arange(18,dtype=torch.float32).reshape(6,1,3)
    positions=torch.tensor([0,2,4]);hours=torch.tensor([400000,400002,400004])
    x=m.causal_features(fields,positions,hours)
    assert x.shape==(3,1,27)
    torch.testing.assert_close(x[1,0,:9],torch.tensor([6.,7.,8.,3.,4.,5.,0.,0.,0.])/10)
    changed=fields.clone();changed[5]+=1000
    torch.testing.assert_close(m.causal_features(changed,positions,hours),x,atol=0,rtol=0)
    assert torch.equal(m.causal_features(torch.zeros_like(fields),positions,hours),torch.zeros_like(x))


def test_normal_equations_fit_known_linear_correction_with_masked_labels():
    from . import innovation_refinement as m
    x=torch.tensor([[[1.,0.],[0.,1.]],[[2.,0.],[0.,2.]]])
    y=torch.tensor([[2.,-1.],[4.,-2.]])
    mask=torch.ones_like(y,dtype=torch.bool);weights=torch.tensor([[1.,3.],[2.,4.]])
    matrix,rhs=m.normal_equations(x,y,mask,weights)
    solution=m.solve_ridge(matrix,rhs,ridge=0)
    torch.testing.assert_close(solution,torch.tensor([2.,-1.],dtype=torch.float64),atol=1e-12,rtol=0)
    y[0,1]=500;mask[0,1]=False
    matrix,rhs=m.normal_equations(x,y,mask,weights)
    torch.testing.assert_close(m.solve_ridge(matrix,rhs,ridge=0),solution,atol=1e-12,rtol=0)
