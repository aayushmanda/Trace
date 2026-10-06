"""python handcoded/one_layer_gradients.py
One-layer softmax attention + linear readout (the model of prop:small-init), counting mod 2.
Exact over all (s0, q, word): ||grad L_O||, its label-free part E[J^T(p-u)], and the
answer-dependent part grad Phi = E[J^T(e_y-u)], versus n and sigma."""
import math, sys, numpy as np, torch
torch.set_default_dtype(torch.float64); dev="cuda"
A, K, d = 4, 2, 16
def run(n, sigma, seed=0):
    V_TOK=A+K+2; N=n+3                     # tokens: letters, K start values, query marker? -> use start, query, letters, answer
    g=torch.Generator(device=dev).manual_seed(seed)
    shapes=dict(E=(A+K+1,d), P=(n+3,d), Wq=(d,d), Wk=(d,d), Wv=(d,d), R=(K,d))
    th={k:(torch.randn(s,generator=g,device=dev)*sigma).requires_grad_() for k,s in shapes.items()}
    words=torch.tensor((np.arange(A**n)[:,None]//A**np.arange(n)[::-1])%A,device=dev)
    lab=0; free=0; tot=0; cnt=0
    G={k:torch.zeros_like(t) for k,t in th.items()}; Gf={k:torch.zeros_like(t) for k,t in th.items()}; Gp={k:torch.zeros_like(t) for k,t in th.items()}
    combos=[(s0,q) for s0 in range(K) for q in range(A)]
    for s0,q in combos:
        m=len(words)
        ids=torch.cat([torch.full((m,1),A+s0,device=dev), torch.full((m,1),q,device=dev), words, torch.full((m,1),A+K,device=dev)],1)
        x=th["E"][ids]+th["P"][None]
        qv=x[:,-1]@th["Wq"].T; kv=x@th["Wk"].T; v=x@th["Wv"].T
        a=torch.softmax((kv@qv.unsqueeze(-1)).squeeze(-1)/math.sqrt(d),-1)
        h=x[:,-1]+(a.unsqueeze(-1)*v).sum(1); z=h@th["R"].T
        y=(s0+(words==q).sum(1))%K; e=torch.eye(K,device=dev)[y]; u=1/K
        p=torch.softmax(z,-1)
        L=torch.nn.functional.cross_entropy(z,y)/len(combos)
        Lf=((p.detach()-u)*z).sum(1).mean()/len(combos)       # grad = E[J^T (p-u)]
        Lp=((e-u)*z).sum(1).mean()/len(combos)                # grad = grad Phi
        for acc,LL in ((G,L),(Gf,Lf),(Gp,Lp)):
            gs=torch.autograd.grad(LL,list(th.values()),retain_graph=True)
            for k,gk in zip(th,gs): acc[k]+=gk
    nrm=lambda D: math.sqrt(sum((t**2).sum().item() for t in D.values()))
    lam=0.5; Ale1=0.5*lam**(2*n)+0.5*n*lam**(2*(n-1))*(1-lam**2)
    return nrm(G), nrm(Gf), nrm(Gp), Ale1
for sigma in [0.01,0.03]:
    for n in [2,4,6,8]:
        g,gf,gp,a=run(n,sigma)
        print(f"sigma={sigma} n={n}  |gradL|={g:.3e}  label-free={gf:.3e} (/sigma^3={gf/sigma**3:.2f})  gradPhi={gp:.3e} (/sigma*sqrtA<=1={gp/(sigma*math.sqrt(a)):.3e})  sqrtA<=1={math.sqrt(a):.3e}")
