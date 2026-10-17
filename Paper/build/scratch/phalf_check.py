import math, numpy as np
from scipy.integrate import solve_ivp
sig=lambda x: 1/(1+math.exp(-x))
def run(eps,dh,q0,k0,v):
    def f(t,y):
        mu,nu,q,k=y
        beta=eps**2*q*k/math.sqrt(dh)
        a0,a1=sig(-beta),sig(beta)
        g0,g1=mu+nu*a0,-mu-nu*a1
        d0,d1=.5*sig(-g0),.5*sig(-g1)
        dLdb=nu*sig(beta)*sig(-beta)*(d0+d1)
        return [eps**2*(d0-d1), eps**2*v**2*(a0*d0-a1*d1), -(eps**2*k/math.sqrt(dh))*dLdb, -(eps**2*q/math.sqrt(dh))*dLdb]
    ev=lambda t,y: (lambda b: (y[0]+y[1]*sig(-b))+(-y[0]-y[1]*sig(b)))(eps**2*y[2]*y[3]/math.sqrt(dh))-4
    ev.terminal=True
    sol=solve_ivp(f,(0,1e12),[0,0,q0,k0],method='LSODA',rtol=1e-10,atol=1e-16,events=ev,dense_output=True)
    T=sol.t_events[0][0]; ts=np.linspace(0,T,4000); Y=sol.sol(ts)
    beta=eps**2*Y[2]*Y[3]/math.sqrt(dh); a0=1/(1+np.exp(beta)); a1=1-a0
    D=Y[0]*2+Y[1]; nu=Y[1]
    bound=258*math.sqrt(dh*math.log(4*math.sqrt(dh)/eps**4))/eps**5+2040/eps**4
    return T, bound, D.max(), np.abs(D).max(), 28.5*eps**2, nu[1:].max(), np.diff(beta).min()
for eps in (0.13,0.1,0.05):
    for dh in (8,64):
        for q0,k0,v in ((eps*1.3,eps*0.8,eps),(eps*0.5,eps*0.5,eps*2),(eps*2,-eps*0.6,eps*0.5),(eps,eps,eps)):
            T,B,Dmax,Dabs,Dbd,numax,dbeta=run(eps,dh,q0,k0,v)
            s=1 if q0*k0>0 else -1
            print(f"eps={eps} dh={dh} sign(beta0)={s:+d}: T*={T:.3e} bound={B:.2e} ok={T<=B}  sign-adjusted maxD={(Dmax if s>0 else -np.min([0])):.1e} |D|max={Dabs:.2e}<= {Dbd:.2e}:{Dabs<=Dbd}  nu sign ok={numax<=0 if s>0 else True} beta monotone={dbeta>=-1e-12 if s>0 else True}")
