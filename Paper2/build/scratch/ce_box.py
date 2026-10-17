import numpy as np
from math import comb, log
from scipy.optimize import brentq
g=0.8
def law(N,M):
    m=N-1-M; us=[];ps=[]
    for k in range(m+1):
        us.append((-M+2*k-m)/(N-1)); ps.append(comb(m,k)/2**m)
    return np.array(us),np.array(ps)
def fj(q,mm,N,us,ps):
    a=np.exp(q)/(np.exp(q)+N-1); s=a+(1-a)*us; t=np.tanh(mm*s/2)
    rho=-(ps*us).sum(); phi=(ps*s).sum()
    f=g*phi-(ps*s*t).sum(); j=a*(1-a)*(ps*(1-us)*(g-t)).sum()
    return f,j
for N in (4,8,16,32,64,128,256):
    q1=log(N-1)
    for M,name in ((0,'k=1'),(1,'k=0')):
        us,ps=law(N,M)
        qs=np.linspace(0,q1,400)
        mstar=[]
        for q in qs:
            fz=lambda mm: fj(q,mm,N,us,ps)[0]
            mstar.append(brentq(fz,1e-9,1e4) if fz(1e-9)>0 and fz(1e4)<0 else 0.0)
        Mbar=max(mstar)
        ms=np.linspace(0,Mbar,400)
        jmin=min(fj(q,mm,N,us,ps)[1]/(np.exp(q)/(np.exp(q)+N-1)*(N-1)/(np.exp(q)+N-1)) for q in qs[::4] for mm in ms[::4])
        jabs=min(fj(q,mm,N,us,ps)[1] for q in qs[::4] for mm in ms[::4])
        print(f"N={N:4d} {name}: Mbar={Mbar:7.3f}  min j/(a(1-a)) over box={jmin:+.4f}  min j={jabs:+.2e}")
