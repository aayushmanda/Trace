import numpy as np
from scipy.integrate import solve_ivp
from mpmath import mp, mpf, log, sqrt
mp.dps=40
# constants
print("712^2",712**2,"15840*32",15840*32,"33/8*3840",33/8*3840,"128*6*5",128*6*5)
ps=np.linspace(1e-6,0.5-1e-9,200001)
hs=np.log((1-ps)/ps)
print("max sqrt(p)h*",(np.sqrt(ps)*hs).max(),2/np.e)
print("max p^2 h*^6",(ps**2*hs**6).max(),(3/np.e)**6)
print("max p h*^4/(1-2p)",(ps*hs**4/(1-2*ps)).max())
sig=lambda z:1/(1+np.exp(-z))
print("sig'(1)",sig(1)*(1-sig(1)),"sig(-1/4)/2",sig(-.25)/2,"tanh(.5)^2",np.tanh(.5)**2,"1/(4sqrt e)",0.25*np.exp(-.5))

def run(K,p,gap_ratio,dh,X,Y,vfac,gap):
    h_star=np.log((1-p)/p)
    eps=np.sqrt(gap*p)*h_star/(712*(1+np.log(K)))
    q0,k0,v=X*eps,Y*eps,vfac*eps
    D=q0**2-k0**2
    assert abs(D)>=gap*eps**2*(1-1e-12)
    N=K-2; rho=2/K
    a0=eps**2*q0*k0/np.sqrt(dh)
    asc=eps**4/np.sqrt(dh)
    mu=p*(1-2*p)/14
    T1=h_star/(2*rho*eps**2*mu)
    Tc=15840*np.sqrt(dh)/(gap*p*h_star*eps**4)
    Tmax=h_star*np.sqrt(dh)/(16*rho*eps**6)
    def phi(m,s):
        A=np.exp(-m); B=N*np.exp(-s-m/2)
        return (A+B/2)/(1+A+B)
    def phis(m,s):  # -dF/ds
        A=np.exp(-m); B=N*np.exp(-s-m/2)
        return B/(1+A+B)
    def rhs(t,y):
        c,g,s,a=y
        r0=sig(-a); r1=sig(a); tau=r1-r0
        m0=c+g*r0; m1=-(c+g*r1)
        e0=(1-p)*phi(m0,s); e1=p*phi(m1,s); err=e0+e1
        Lc=e1-e0; Lg=r1*e1-r0*e0; La=g*sig(a)*(1-sig(a))*err
        Ls=-((1-p)*phis(m0,s)+p*phis(m1,s))
        S=np.sqrt(D**2+4*dh*a**2/eps**4)
        ds=-(eps**2/(2*N))*Ls if N>0 else 0.0
        return [-rho*eps**2*Lc,-rho*eps**2*v**2*Lg,ds,-eps**4*S/dh*La]
    ev=lambda t,y:y[3]+asc/8
    ev.terminal=True
    sol=solve_ivp(rhs,[0,Tmax],[0,0,0,a0],method='LSODA',rtol=1e-10,atol=[1e-14,1e-20,1e-14,1e-24],events=ev,dense_output=False,max_step=Tmax/2000)
    ta=sol.t_events[0][0] if len(sol.t_events[0]) else np.inf
    c,g,s,a=sol.y
    h=c+g/2; Q=g-v**2/2*c
    Delta=-g*np.tanh(a/2)
    print(f"K={K} p={p} gap={gap} dh={dh} X,Y={X:.3f},{Y:.3f} v/eps={vfac} eps={eps:.3e}: t_a={ta:.3e}, bound T1+Tc={T1+Tc:.3e}, ratio={ta/(T1+Tc):.3e}; min g={g.min():.2e}, max|Q|/(v^2h*/16)={np.abs(Q).max()/(v**2*h_star/16):.3e}, min h={h.min():.2e}, max|Delta|={np.abs(Delta).max():.2e}")
for K in [2,3,6]:
  for p in [0.05,0.25,0.45]:
    for (X,Y,gap) in [(2.0,0.5,1.0),(1.3,0.8,0.5),(1.0,np.sqrt(1-0.05),0.05)]:
      run(K,p,None,4,X,Y,0.5,gap)
