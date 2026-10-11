"""Original research prototype: learn branch outcome differences with masked anchors.

No claim of novel theorem or performance improvement is made. Requires torch.
"""
from __future__ import annotations
import torch
from torch import nn

class VisualBranchModel(nn.Module):
    """Encode RGB history and action, predict event logit and displacement."""
    def __init__(self, action_dim=3, width=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.GELU(),
            nn.Conv2d(16,32,3,2,1),nn.GELU(),
            nn.Conv2d(32,width,3,2,1),nn.GELU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(width+action_dim,width),nn.GELU(),
                                nn.Linear(width,2))
    def forward(self, rgb, actions):
        # rgb [B,3,H,W], actions [B,K,D]
        b,k,_=actions.shape
        state=self.encoder(rgb).unsqueeze(1).expand(-1,k,-1)
        return self.head(torch.cat((state,actions),dim=-1))

def objective(outputs, event, displacement, mask, edges=None, pair_weight=0.0):
    """Masked outcome loss plus same-state cross-action *difference* supervision.

    outputs: [B,K,2]; event, displacement, mask: [B,K].
    edges: list of (i,j) action pairs; both must be observed for supervision.
    """
    import torch.nn.functional as F
    if outputs.ndim!=3 or outputs.shape[-1]!=2:raise ValueError("outputs [B,K,2]")
    if mask.shape!=event.shape or event.shape!=outputs.shape[:2]:
        raise ValueError("invalid shapes")
    observed=mask.to(outputs.dtype)
    denom=observed.sum().clamp_min(1)
    loss=((F.binary_cross_entropy_with_logits(outputs[...,0],event.float(),reduction="none")
         +F.smooth_l1_loss(outputs[...,1],displacement.float(),reduction="none"))*observed).sum()/denom
    if edges and pair_weight:
        paired=outputs.new_zeros(())
        count=outputs.new_zeros(())
        for i,j in edges:
            valid=observed[:,i]*observed[:,j]
            predicted=outputs[:,i,1]-outputs[:,j,1]
            actual=displacement[:,i]-displacement[:,j]
            paired=paired+(F.smooth_l1_loss(predicted,actual,reduction="none")*valid).sum()
            count=count+valid.sum()
        loss=loss+pair_weight*paired/count.clamp_min(1)
    return loss

def train_epoch(model,loader,optimizer,device="cpu",pair_weight=0.0):
    model.train();total=0.;steps=0
    for batch in loader:
        rgb,actions,event,displacement,mask=[v.to(device) for v in batch]
        optimizer.zero_grad(set_to_none=True)
        prediction=model(rgb,actions)
        edges=[(i,i+1) for i in range(actions.shape[1]-1)]
        loss=objective(prediction,event,displacement,mask,edges,pair_weight)
        loss.backward();optimizer.step()
        total+=float(loss.detach());steps+=1
    return total/max(steps,1)

@torch.no_grad()
def evaluate(model,loader,device="cpu",boundary_margin=0.05):
    model.eval();n=0;correct=0;brier=0.;boundary_n=0;boundary_correct=0
    for batch in loader:
        rgb,actions,event,displacement,mask=[v.to(device) for v in batch]
        p=torch.sigmoid(model(rgb,actions)[...,0]);pred=(p>=0.5)
        observed=mask.bool();correct+=int(((pred==event.bool())&observed).sum())
        n+=int(observed.sum())
        brier+=float((((p-event)**2)*observed).sum())
        # Only a proxy: boundary defined by observed displacement near zero.
        near=(displacement.abs()<boundary_margin)&observed
        boundary_n+=int(near.sum())
        boundary_correct+=int(((pred==event.bool())&near).sum())
    return {"accuracy":correct/max(n,1),"brier":brier/max(n,1),
            "near_zero_displacement_accuracy":boundary_correct/max(boundary_n,1),
            "n":n,"near_zero_n":boundary_n}
