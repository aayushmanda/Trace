"""Untied one-block population training beyond the restricted theorem.

Fixed seven-dimensional embeddings and query type mask; all query/key/value,
attention output, MLP and readout matrices train from seeded normal weights.
No layer normalization/dropout, SGD, exact noisy population expectations.
Run with the repository Python environment. Results are exploratory controls,
not claims of unrestricted-model scaling.
"""
from pathlib import Path
import csv
import math
import torch
from one_block import shared_population as population

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]

class UntiedBlock(torch.nn.Module):
    def __init__(self, scale, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.query=torch.nn.Linear(7,7,bias=False)
        self.key=torch.nn.Linear(7,7,bias=False)
        self.value=torch.nn.Linear(7,7,bias=False)
        self.output=torch.nn.Linear(7,7,bias=False)
        self.mlp=torch.nn.Sequential(torch.nn.Linear(7,16),torch.nn.ReLU(),torch.nn.Linear(16,7))
        self.readout=torch.nn.Linear(7,1,bias=False)
        with torch.no_grad():
            for name,p in self.named_parameters():
                if 'bias' in name: p.zero_()
                else: p.normal_(std=scale)
    def forward(self,e):
        query=e[:,-1]
        q=self.query(query)
        keys=self.key(e[:,:-1])
        a=torch.softmax((keys*q[:,None]).sum(-1)/math.sqrt(7),dim=1)
        h=query+self.output((a[:,:,None]*self.value(e[:,:-1])).sum(1))
        h=h+self.mlp(h)
        return self.readout(h).squeeze(-1),a[:,-1]

def run(n,k,u,seed,steps=4000,lr=.2):
    x,w,b,y,p=population(n,k)
    e=torch.zeros(x.numel(),n+1,7)
    e[:,:n-1,0]=w[:,None];e[:,n-1,0]=x;e[:,:,2]=1
    e[:,:n-1,4]=1;e[:,n-1,3]=1;e[:,n,1]=b
    model=UntiedBlock(u,seed)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr)
    clean=b*x
    hit_accuracy=hit_attention=None
    for step in range(steps+1):
        z,a=model(e)
        loss=(p*torch.nn.functional.softplus(-y*z)).sum()
        accuracy=float((p*(z.detach()*clean>0)).sum())
        attention=float((p*a.detach()).sum())
        if hit_accuracy is None and accuracy>=.99: hit_accuracy=step
        if hit_attention is None and float(a.detach().min())>=.5:hit_attention=step
        if step==steps:break
        optimizer.zero_grad();loss.backward();optimizer.step()
    return dict(n=n,k=k,u=u,seed=seed,steps=steps,lr=lr,clean_accuracy=accuracy,ce_risk=float(loss.detach()),mean_source_attention=attention,min_source_attention=float(a.detach().min()),accuracy_hit=hit_accuracy,route_hit=hit_attention)

def main():
    results=[]
    for n in (16,64):
        for u in (.1,.03):
            for seed in (0,1,2):
                for k in (1.,0.):
                    row=run(n,k,u,seed);results.append(row)
                    print(row,flush=True)
    out=ROOT/'checks'/'results';out.mkdir(exist_ok=True)
    with (out/'untied_one_block.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=results[0].keys());wr.writeheader();wr.writerows(results)
    lines=[r'\begin{tabular}{@{}rrr rrr@{}}',r'\toprule',r'$N$ & $\varepsilon$ & $\kappa$ & clean accuracy & cross-entropy & mean source attention\\',r'\midrule']
    for n in (16,64):
        for u in (.1,.03):
            for k in (1.,0.):
                rows=[r for r in results if (r['n'],r['u'],r['k'])==(n,u,k)]
                def ms(key):
                    vals=torch.tensor([r[key] for r in rows]);return f'{float(vals.mean()):.3f} $\\pm$ {float(vals.std(unbiased=False)):.3f}'
                lines.append(f'{n} & {u:g} & {k:g} & '+ ' & '.join(ms(key) for key in ('clean_accuracy','ce_risk','mean_source_attention'))+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'figures'/'untied_table.tex').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
