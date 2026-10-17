from model import *
import sys
n=int(sys.argv[1]); iters=int(sys.argv[2]); seed=int(sys.argv[3]); eps=0.3
out,st=data(n,.25); rng=np.random.default_rng(seed); P=init(n,eps,rng); W=len(P['v'])
P['wu']=eps*rng.normal(size=W)*3; P['d']=eps*rng.normal(size=W); P['b']=eps*0.1*rng.normal(size=W)
hit=None
for it in range(iters):
    L,g,acc,mm=loss_grad(P,out)
    if acc>=0.9: hit=it;break
    for k in P: P[k]=P[k]-g[k]
print('std',n,seed,'outcome iters to 90%',hit,'acc',round(acc,4))
