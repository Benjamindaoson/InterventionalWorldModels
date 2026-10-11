import pytest
torch=pytest.importorskip("torch")
from iwm.learning import VisualBranchModel,objective
def test_forward_backward():
    model=VisualBranchModel(action_dim=3,width=16)
    out=model(torch.rand(2,3,64,64),torch.rand(2,4,3))
    assert out.shape==(2,4,2)
    loss=objective(out,torch.zeros(2,4),torch.ones(2,4),torch.ones(2,4),[(0,1)],0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
def test_mask():
    out=torch.zeros(1,2,2);mask=torch.tensor([[1.,0.]])
    a=objective(out,torch.zeros(1,2),torch.zeros(1,2),mask)
    b=objective(out,torch.tensor([[0.,1.]]),torch.tensor([[0.,100.]]),mask)
    assert torch.allclose(a,b)
