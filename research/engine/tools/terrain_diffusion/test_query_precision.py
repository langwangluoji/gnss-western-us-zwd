import torch


def test_query_local_time_fusion_preserves_reference_and_query_set_invariance():
    from .p14_fusion import P14Fusion
    from .query_precision import QueryResidual
    torch.manual_seed(312);old=P14Fusion('gru',12)
    model=QueryResidual(old.state_dict(),'gru',length=12,side=4,width=16).eval()
    B,N=2,5;uv=torch.rand(B,1,N,2)*2-1
    b=dict(point=torch.randn(B,N,113),support=torch.randn(B,30,8,8),uv=uv,shape=(8,8),
        static=torch.randn(B,N,9),grid_static=torch.randn(64,8),edges=torch.randn(B,64,3,10),
        edge_mask=torch.ones(B,64,3),history=torch.randn(B,12,16,25),time_static=torch.randn(16,8),
        legacy_history=torch.randn(B,12,64,25),time_uv=uv[:,0],calendar=torch.randn(B,4))
    with torch.no_grad():
        torch.testing.assert_close(model(b),model.reference(b),atol=0,rtol=0)
        model.local[-1].weight.normal_(0,.1)
        allq=model(b);small=dict(b,point=b['point'][:,:2],static=b['static'][:,:2],uv=uv[:,:,:2],time_uv=b['time_uv'][:,:2])
        torch.testing.assert_close(model(small),allq[:,:2],atol=1e-6,rtol=1e-5)
