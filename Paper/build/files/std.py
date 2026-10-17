from model import *
import sys
n=int(sys.argv[1]); eps=float(sys.argv[2]); iters=int(sys.argv[3]); seed=int(sys.argv[4]); mode=sys.argv[5]
out,st=data(n,.25); rng=np.random.default_rng(seed); P=init(n,eps,rng); W=len(P['v'])
if mode=='std':  # default-like: kinks inside the input range
    P['wu']=eps*rng.normal(size=W)*3; P['d']=eps*rng.normal(size=W); P['b']=eps*0.1*rng.normal(size=W)
for D,name in ((st,'step'),(out,'out')):
    Q={k:(np.copy(v) if isinstance(v,np.ndarray) else v) for k,v in P.items()}
    hit=None
    for it in range(iters):
        L,g,acc,mm=loss_grad(Q,D)
        if mm>0: hit=it;break
        for k in Q: Q[k]=Q[k]-g[k]
    print(mode,n,seed,name,'solve',hit,'acc',round(acc,4),'L',round(L,4))
