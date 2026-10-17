from model import *
from run import flat
import sys
n=int(sys.argv[1]); eps=float(sys.argv[2]); iters=int(sys.argv[3]); seed=int(sys.argv[4])
out,st=data(n,.25); P=init(n,eps,np.random.default_rng(seed))
th0=flat(P)
for it in range(iters+1):
    L,g,acc,mm=loss_grad(P,out)
    if it in (0,100,1000,10000,100000,200000,400000) or it==iters:
        u=np.linspace(0,1,201)
        kink_in = 0
        for sp in (-1,1):
            Z=np.outer(u,P['wu'])+sp*P['d']+P['b']
            kink_in += np.sum((Z.max(0)>0)&(Z.min(0)<0))
        gn=np.sqrt(sum(np.sum(np.square(v)) for v in g.values()))
        print(it,'L',round(L,6),'acc',round(acc,4),'|grad|',f'{gn:.2e}','disp/eps',round(np.linalg.norm(flat(P)-th0)/eps,3),'units with kink in range',kink_in, 'target acc',(1+.5**(n-1))/2)
    for k in P: P[k]=P[k]-g[k]
