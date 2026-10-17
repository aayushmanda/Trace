"""Matrix-level one-block realization and exact population-flow experiments.

Uses fixed embeddings, one scalar query/key head, residuals and a four-unit
ReLU MLP. Only tied scalar directions are trained, as defined in the paper.
Law D_kappa: M anti-copies of the source (E[M] = 1 - kappa) at random decoy positions,
the remaining decoys iid uniform. No sampling: every (source, anti-copy count, number of
agreeing iid decoys, instruction, error) row is enumerated with its exact mass in float64;
decoys share a score, so one ordering per row gives the exact logit and gradient.
Run: python Paper2/checks/one_block.py
"""
from pathlib import Path
import csv
import itertools
import json
import math
import numpy as np
import torch
from scipy.integrate import solve_ivp
from anti_law import anti_mix, ce_field, u_law
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]

def shared_population(n, k, p=.1):
    """Former two-state law (one shared decoy value w); kept for the legacy untied script."""
    d = (1-k)/(n-1)
    rows = []
    for x, w, b, e in itertools.product((-1., 1.), repeat=4):
        weight = (1-d*x*w)/8 * (p if e == -1 else 1-p)
        rows.append((x, w, b, b*x*e, weight))
    return torch.tensor(rows).T

def population(n, k, p=.1):
    """Rows of the anti-copy law: decoy matrix X (rows, n-1), source x, instruction b, label y, mass."""
    X, xs, bs, ys, ws = [], [], [], [], []
    for M, pm in anti_mix(k):
        free = n-1-M
        for x, j, b, e in itertools.product((-1., 1.), range(free+1), (-1., 1.), (-1., 1.)):
            X.append([-x]*M + [x]*j + [-x]*(free-j))
            xs.append(x); bs.append(b); ys.append(b*x*e)
            ws.append(pm*math.comb(free, j)/2**free/4 * (p if e == -1 else 1-p))
    T = torch.tensor
    return T(X), T(xs), T(bs), T(ys), T(ws)

class OneBlock(torch.nn.Module):
    def __init__(self, scores=(0., 0.), factors=(.01, .01), bound=32.):
        super().__init__()
        self.qs = torch.nn.Parameter(torch.tensor(scores[0]))
        self.qd = torch.nn.Parameter(torch.tensor(scores[1]))
        self.r1 = torch.nn.Parameter(torch.tensor(factors[0]))
        self.r2 = torch.nn.Parameter(torch.tensor(factors[1]))
        self.bound = bound

    def forward(self, x, decoys, b, n):
        batch = x.numel()
        # Explicit sequence embeddings: [value, instruction, constant, source,
        # decoy, attended value, MLP output]. Final query sees only earlier states.
        e = torch.zeros(batch, n+1, 7)
        e[:, :n-1, 0] = decoys
        e[:, n-1, 0] = x
        e[:, :, 2] = 1
        e[:, :n-1, 4] = 1
        e[:, n-1, 3] = 1
        e[:, n, 1] = b
        query_map = torch.tensor([0., 0., 1., 0., 0., 0., 0.])
        key_map = torch.stack((self.qs*0, self.qs*0, self.qs*0, self.qs, self.qd, self.qs*0, self.qs*0))
        value_map = torch.tensor([1., 0., 0., 0., 0., 0., 0.])
        query = e[:, n] @ query_map
        keys = e[:, :n] @ key_map
        a = torch.softmax(query[:, None]*keys, dim=1)
        v = (a * (e[:, :n] @ value_map)).sum(dim=1)
        residual = e[:, n].clone()
        residual[:, 5] += v
        # Four first-layer rows, applied as a matrix; signs tie both layers.
        signs = torch.tensor([1., -1., 1., -1.])
        instr = torch.tensor([1., 1., -1., -1.]) * self.bound
        rows = torch.zeros(4, 7)
        rows[:, 1] = instr
        rows[:, 5] = signs * self.r1
        pre = residual @ rows.T - self.bound
        # Coherent half-subgradient at zero: ReLU(t)-ReLU(-t) differentiates to 1.
        hidden = torch.where(pre > 0, pre, torch.where(pre < 0, pre*0, pre*.5))
        output_rows = self.r2 * torch.tensor([1., -1., -1., 1.])
        z = residual[:, 6] + hidden @ output_rows
        return z, a[:, -1]

def coefficients(n, k, c, s, m, loss):
    """Rule-product and score-gap coefficients: dm/dt = m A, dq/dt = m B (balanced flow)."""
    d = (1-k)/(n-1)
    a = 1/(1+(n-1)*math.exp(-s))
    if loss == 'correlation':
        return c*((1+d)*a-d), c*(1+d)*a*(1-a)
    us, ps = u_law(n, k)
    return ce_field(s, m, n, us, ps, c)

def check_network():
    max_logit = max_grad = 0.
    for n, k, loss in itertools.product((4, 16, 64), (1., 0., -.2, -1.), ('correlation', 'ce')):
        X,x,b,y,p = population(n,k,.1)
        for s,r in ((0., .01),(.2,.1),(math.log(n-1),.7)):
            net = OneBlock(scores=(s/2,-s/2), factors=(r,r))
            z,a = net(x,X,b,n)
            aa = 1/(1+(n-1)*math.exp(-s))
            expected = r*r*b*(aa*x+(1-aa)*X.mean(1))
            err = (z-expected).abs().max().item()
            max_logit = max(max_logit,err)
            L = (p * (-y*z/2 if loss=='correlation' else torch.nn.functional.softplus(-y*z))).sum()
            L.backward()
            A,B = coefficients(n,k,.8,s,r*r,loss)
            observed = np.array([-net.r1.grad.item(),-net.r2.grad.item(),-net.qs.grad.item()+net.qd.grad.item()])
            theory = np.array([r*A/2,r*A/2,r*r*B])
            max_grad = max(max_grad,float(np.max(np.abs(observed-theory))))
    assert max_logit < 1e-12 and max_grad < 1e-12
    return {'max_logit_error':max_logit,'max_gradient_error':max_grad,'configurations':72}

def integrate(n,k,u,loss,gap=None):
    target = math.log(n-1) if gap is None else gap
    def rhs(t, state):
        m,s = state
        A,B = coefficients(n,k,.8,s,m,loss)
        return [m*A,m*B]
    def stop(t,state): return state[1]-target
    stop.terminal=True
    stop.direction=1
    sol = solve_ivp(rhs,(0,1e8),(u*u,0),events=stop,rtol=2e-10,atol=1e-15,method='DOP853')
    if not len(sol.t_events[0]): raise RuntimeError((n,k,u,loss,'no route'))
    time = float(sol.t_events[0][0])
    m,s = sol.y[:,-1]
    a = 1/(1+(n-1)*math.exp(-s))
    X,x,b,y,p = population(n,k)
    z = m*b*(a*x+(1-a)*X.mean(1))
    risk = float((p*torch.nn.functional.softplus(-y*z)).sum())
    return time,risk,float(m)

def render_results(records):
    lines=[r'\begin{tabular}{@{}rr rrrr@{}}',r'\toprule',r'$N$ & $\varepsilon$ & $T_{\mathrm{corr},1}$ & $T_{\mathrm{corr},0}$ & $T_{\mathrm{ce},1}$ & $T_{\mathrm{ce},0}$\\',r'\midrule']
    for n,u in itertools.product((16,64),(.03,.01,.003,.001)):
        vals=[next(r['route_time'] for r in records if (r['n'],r['u'],r['loss'],r['k'])==(n,u,loss,k)) for loss,k in itertools.product(('correlation','ce'),(1.,0.))]
        lines.append(f'{n} & {u:g} & '+ ' & '.join(f'{v:,.1f}' for v in vals)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'figures'/'one_block_table.tex').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(8,3.1))
    for ax,key,title in zip(axs,('escape_time','route_time'),('Local escape: score gap 0.01','Routing: source attention 0.5')):
        for loss,style in (('correlation','--'),('ce','-')):
            for k,color in ((1.,'#174f94'),(0.,'#b22e38')):
                rr=[r for r in records if (r['n'],r['loss'],r['k'])==(16,loss,k)]
                ax.loglog([r['u'] for r in rr],[r[key] for r in rr],style+'o',color=color,label=fr'{loss}, $\kappa={k:g}$')
        ax.set_xlabel(r'initialization $\varepsilon$'); ax.set_ylabel('gradient-flow time'); ax.set_title(title,fontsize=10); ax.grid(alpha=.2)
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT/'figures'/'one_block_scaling.pdf'); plt.close(fig)
    return lines

def main():
    out = ROOT/'checks'/'results'
    out.mkdir(exist_ok=True)
    audit = check_network()
    records=[]
    for n,k,u,loss in itertools.product((16,64), (1.,0.), (.03,.01,.003,.001,.0003,.0001), ('correlation','ce')):
        t,risk,m = integrate(n,k,u,loss)
        tb,_,_ = integrate(n,k,u,loss,gap=.01)
        records.append(dict(n=n,k=k,u=u,loss=loss,route_time=t,escape_time=tb,ce_risk_at_route=risk,m_at_route=m))
    with (out/'one_block.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=records[0].keys()); wr.writeheader(); wr.writerows(records)
    audit['flow_runs']=len(records)*2
    (out/'one_block_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines = render_results(records)
    print(json.dumps(audit,indent=2))
    print('\n'.join(lines))

if __name__=='__main__': main()
