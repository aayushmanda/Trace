import sys, math, torch
sys.argv=[sys.argv[0],"--device","cpu"]
sys.path.insert(0,"/home/hariguru/aayus/trace/Paper2/checks")
import untied_ladder as U
torch.set_default_dtype(torch.float64)
N,ETA,B=16,.1,4.
def point(qs,qd,r):
    th=dict(Wq=torch.zeros(7,7),Wk=torch.zeros(7,7),Wv=torch.zeros(7,7),Wo=torch.zeros(7,7),W1=torch.zeros(4,7),b1=torch.zeros(4),W2=torch.zeros(7,4),u=torch.zeros(7))
    th["Wq"][0,2]=th["Wv"][0,0]=th["Wo"][5,0]=th["u"][6]=1.
    th["Wk"][0,3]=qs; th["Wk"][0,4]=qd
    th["W1"][:4,5]=r*torch.tensor([1.,-1.,1.,-1.]); th["W1"][:4,1]=B*torch.tensor([1.,1.,-1.,-1.]); th["b1"][:]=-B
    th["W2"][6,:4]=r*torch.tensor([1.,-1.,-1.,1.])
    return {k:v[None].clone().requires_grad_() for k,v in th.items()}
dirs={"Wk":[(0,3),(0,4)]}
for m,kappa in ((0,1.),(1,0.)):
    e,y,_,_,_,P=U.law_rows("anti",m,N,ETA)
    for qs,qd,r in ((0,0,.01),(0,0,.1),(.3,-.3,.1),(1.,-1.,.5)):
        th=point(qs,qd,r)
        z=U.forward(th,e,N)[0]
        L=(P*torch.nn.functional.softplus(-y*z)).sum()
        g=torch.autograd.grad(L,list(th.values()))
        G={k:gi[0].clone() for k,gi in zip(th,g)}
        tied=G["Wk"][0,3]**2+G["Wk"][0,4]**2
        # project W1 rule column and W2 rule row onto their tied patterns
        p1=torch.tensor([1.,-1.,1.,-1.])/2; p2=torch.tensor([1.,-1.,-1.,1.])/2
        c1=(G["W1"][:4,5]@p1); c2=(G["W2"][6,:4]@p2)
        G["W1"][:4,5]-=c1*p1; G["W2"][6,:4]-=c2*p2; G["Wk"][0,3]=0; G["Wk"][0,4]=0
        off=math.sqrt(sum(float((v**2).sum()) for v in G.values()))
        on=math.sqrt(float(tied+c1**2+c2**2))
        big=max(((k,float(v.abs().max())) for k,v in G.items()),key=lambda t:t[1])
        print(f"kappa={kappa:+.0f} q=({qs},{qd}) r={r}: on-manifold grad {on:.3e}  off-manifold {off:.3e}  largest off block {big}")
