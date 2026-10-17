import numpy as np, itertools
sig=lambda x: 1/(1+np.exp(-x))
def data(n,p):
    B=np.array(list(itertools.product([0,1],repeat=n)))
    w=np.prod(np.where(B==1,p,1-p),1)
    B=np.concatenate([B,B]); w=np.concatenate([w,w])/2
    s0=np.r_[np.zeros(len(w)//2),np.ones(len(w)//2)].astype(int)
    S=(s0[:,None]+np.cumsum(B,1))%2            # s_1..s_n
    sg=lambda s:1-2*s
    # outcome: answer position; displayed state = s0, gate letter = last, avg over all letters
    out=dict(sp=sg(s0),chi=sg(B[:,-1]),u=B.mean(1),Y=sg(S[:,-1]),w=w)
    # step: positions t=2..n
    sp,ch,u,Y,ww=[],[],[],[],[]
    for t in range(2,n+1):
        sp.append(sg(S[:,t-2])); ch.append(sg(B[:,t-1])); u.append(B[:,:t].mean(1))
        Y.append(sg(S[:,t-1])); ww.append(w/(n-1))
    st=dict(sp=np.concatenate(sp),chi=np.concatenate(ch),u=np.concatenate(u),Y=np.concatenate(Y),w=np.concatenate(ww))
    return out,st
def fwd(P,D):
    T=np.tanh(P['q']*P['k'])
    Z=np.outer(D['u'],P['wu'])+np.outer(D['sp'],P['d'])+P['b']
    H=np.maximum(Z,0)
    f=D['sp']*(P['al']+P['be']*T*D['chi'])+H@P['v']
    return f,T,Z,H
def loss_grad(P,D):
    f,T,Z,H=fwd(P,D); Y,w=D['Y'],D['w']
    L=np.sum(w*np.logaddexp(0,-Y*f))
    G=w*(-Y*sig(-Y*f))
    g={}
    g['al']=np.sum(G*D['sp']); g['be']=np.sum(G*D['sp']*T*D['chi'])
    dT=np.sum(G*D['sp']*P['be']*D['chi'])*(1-T**2)
    g['q']=dT*P['k']; g['k']=dT*P['q']
    g['v']=G@H
    A=(Z>0)*(G[:,None])*P['v']
    g['wu']=D['u']@A; g['d']=D['sp']@A; g['b']=A.sum(0)
    acc=np.sum(w*(Y*f>0))
    return L,g,acc,np.min(Y*f)
def init(n,eps,rng):
    W=6*(n+1)
    return dict(al=0.,be=0.,q=eps*rng.uniform(.5,2)*rng.choice([-1,1]),k=eps*rng.uniform(.5,2)*rng.choice([-1,1]),
        v=np.zeros(W),wu=eps*0.2*rng.normal(size=W)/np.sqrt(2),d=eps*0.2*rng.normal(size=W)/np.sqrt(2),
        b=eps*rng.uniform(1,2,W)*rng.choice([-1,1],W))
def outcome_executor(n,A=20.):
    M=n+2; wu=[];d=[];b=[];v=[]
    for a in range(n+1):
        for s in (1,-1):
            for off,coef in ((1,1),(0,-2),(-1,1)):
                # ReLU(n u - a + off + M(s*sp - 1))
                wu.append(n); d.append(M*s); b.append(-a+off-M); v.append(A*s*(-1)**a*coef)
    return dict(al=0.,be=0.,q=0.,k=0.,v=np.array(v,float),wu=np.array(wu,float),d=np.array(d,float),b=np.array(b,float))
def step_executor(n,W):
    return dict(al=0.,be=20.,q=5.,k=5.,v=np.zeros(W),wu=np.zeros(W),d=np.zeros(W),b=-np.ones(W))
