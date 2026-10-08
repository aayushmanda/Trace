"""Numerical check of the staged routing theorem: exact population over all letter patterns and both
start counts, segment mask, stages A (prior, exact minimizer), B (one step on v_r), C (one step on beta, scaled so the
current-letter bias reaches L0), D (gradient descent on the readout). Usage: python routing_stages_check.py n p a g lr [steps]
"""
import itertools, math, sys
import torch
torch.set_default_dtype(torch.float64)
LR=float(sys.argv[5]); n=int(sys.argv[1]); p=float(sys.argv[2]); a=float(sys.argv[3]); g=float(sys.argv[4]); sig=0.1
pats=torch.tensor(list(itertools.product([1.,-1.],repeat=n)))       # m_j
w=torch.prod(torch.where(pats<0,torch.tensor(p),torch.tensor(1-p)),1)
x0s=torch.tensor([1.,-1.])
m=pats.repeat(2,1); x0=x0s.repeat_interleave(len(pats)); wt=w.repeat(2)/2
x=torch.cat([x0[:,None], x0[:,None]*m[:,:-1].cumprod(1)],1)          # x_t, t=1..n
y=x*m
ts=torch.arange(1,n)                                                   # t=2..n
nb=2*n+2
def offs(t):
    return torch.tensor([n + t - j for j in range(n)])
beta=torch.zeros(nb,requires_grad=True)
vc=torch.zeros(1,requires_grad=True); vs=torch.zeros(1,requires_grad=True); vr=torch.zeros(1,requires_grad=True)
def Mt(t):
    sc=a*x[:,t:t+1]*m + beta[offs(t)][None,:]
    num=torch.exp(sc).sum(1); return num/(num+torch.exp(beta[0]))
def loss():
    L=0
    for t in ts:
        f=sig*vc+sig*vs*x[:,t]+g*vr*Mt(t)
        L=L+(wt*torch.nn.functional.softplus(-y[:,t]*f)).sum()
    return L/len(ts)
def margins():
    with torch.no_grad():
        return torch.stack([y[:,t]*(sig*vc+sig*vs*x[:,t]+g*vr*Mt(t)) for t in ts]).min().item()
s=math.log((1-p)/p)
with torch.no_grad(): vs.fill_(s/sig)
lossv=loss(); lossv.backward()
etaB=1.0/(g*g)
with torch.no_grad(): vr-= etaB*vr.grad
theta=math.copysign(1,a)*g*vr.item(); print("Stage B theta=",theta)
for q in (vc,vs,vr,beta): q.grad=None
loss().backward(); G=beta.grad.clone()
L0=12.0
etaC=L0/(-G[n].item())
with torch.no_grad(): beta-= etaC*G
print(f"beta_n={beta[n].item():.3f} beta_0={beta[0].item():.3f} max other={max(beta[d].item() for d in range(1,nb) if d!=n):.3f}")
opt=[vc,vs,vr]
STEPS=int(sys.argv[6]) if len(sys.argv)>6 else 200000
for it in range(STEPS+1):
    for q in opt: q.grad=None
    l=loss(); l.backward()
    with torch.no_grad():
        for q in opt: q-=LR*q.grad
    if it%max(STEPS//10,1)==0:
        mg=margins(); print(f"D-step {it} loss {l.item():.4e} min margin {mg:+.4f}")
        if mg>0: print("all prompts, all t>=2: positive margin"); break
