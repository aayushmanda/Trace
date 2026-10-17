import math
g=0.8
def run(N,kappa,eps,h=1e-4,tmax=50):
    rho=(1-kappa)/(N-1); qs=qd=0.; r1=r2=eps; t=0.
    sgn=lambda x: (x>0)-(x<0)
    while t<tmax:
        q=qs-qd; a=math.exp(q)/(math.exp(q)+N-1)
        phi=(1+rho)*a-rho; dphi=(1+rho)*a*(1-a)
        if a>=0.5: return t
        # L = -(g/2) r1 r2 phi(q)
        G=(-(g/2)*r2*phi, -(g/2)*r1*phi, -(g/2)*r1*r2*dphi, (g/2)*r1*r2*dphi)
        if all(abs(x)<1e-300 for x in G): return None
        r1-=h*sgn(G[0]); r2-=h*sgn(G[1]); qs-=h*sgn(G[2]); qd-=h*sgn(G[3]); t+=h
    return None
for N in (16,64):
    print(N,'predicted 0.5*log(N-1)=',round(0.5*math.log(N-1),4))
    for kappa in (1.,0.):
        print('  kappa',kappa,[None if (x:=run(N,kappa,e)) is None else round(x,3) for e in (0.1,0.01,0.001)])
    for kappa,eps in ((-0.2,0.05),(-0.2,0.15),(-1.,0.3),(-1.,0.4)):
        thr=0.5*math.log(1-kappa); print('  kappa',kappa,'eps',eps,'threshold',round(thr,4),'route',run(N,kappa,eps))
