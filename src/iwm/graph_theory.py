"""Deterministic graph error certificate (not neural generalization)."""
import numpy as np
def certificate(errors,edges,source,target):
    e=np.asarray(errors,dtype=float);n=len(e)
    B=np.zeros((len(edges),n))
    for k,(i,j) in enumerate(edges):
        if i==j or not(0<=i<n and 0<=j<n):raise ValueError("invalid edge")
        B[k,i]=1;B[k,j]=-1
    seen={source};stack=[source]
    while stack:
        u=stack.pop()
        for i,j in edges:
            v=j if i==u else (i if j==u else None)
            if v is not None and v not in seen:seen.add(v);stack.append(v)
    if target not in seen:return float("inf")
    L=B.T@B;d=np.eye(n)[target]-np.eye(n)[source]
    resistance=max(0.,float(d@np.linalg.pinv(L)@d))
    return abs(float(e[source]))+np.sqrt(resistance*float(np.sum((B@e)**2)))
