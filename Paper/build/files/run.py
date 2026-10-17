from model import *
import sys,time,json
def flat(P): return np.concatenate([np.atleast_1d(P[k]) for k in sorted(P)])
def train(P,D,lr,iters,eps,log_every=0):
    P={k:(np.copy(v) if isinstance(v,np.ndarray) else v) for k,v in P.items()}
    th0=flat(P); t_exit=t_solve=t_drop=None; L0=None; hist=[]
    for it in range(iters):
        L,g,acc,mm=loss_grad(P,D)
        if L0 is None: L0=L
        if t_exit is None and np.linalg.norm(flat(P)-th0)>=0.5*eps: t_exit=it
        if t_drop is None and L<=np.log(2)-0.05: t_drop=it
        if t_solve is None and mm>0: t_solve=it; break
        if log_every and it%log_every==0: hist.append((it,L,acc,P['be']*np.tanh(P['q']*P['k'])))
        for k in P: P[k]=P[k]-lr*g[k]
    return dict(exit=t_exit,drop=t_drop,solve=t_solve,L=L,acc=acc,hist=hist)
if __name__=='__main__':
    n=int(sys.argv[1]); eps=float(sys.argv[2]); iters=int(sys.argv[3]); seed=int(sys.argv[4])
    out,st=data(n,.25); rng=np.random.default_rng(seed); P=init(n,eps,rng)
    t=time.time(); rs=train(P,st,1.0,iters,eps); ro=train(P,out,1.0,iters,eps)
    rs.pop('hist');ro.pop('hist')
    print(json.dumps(dict(n=n,seed=seed,step=rs,out=ro,sec=round(time.time()-t,1))))
