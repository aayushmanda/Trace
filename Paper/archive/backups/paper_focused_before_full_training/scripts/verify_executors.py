"""Exhaustive gold-prefix and free-rollout verification of finite softmax executors."""
import itertools,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'verification'))
import lettertrace as L
import executor_check as independent

torch.set_num_threads(1)
rows=[]
with torch.inference_mode():
    for n,A,K in [(n,A,K) for n in range(2,7) for A in [2,3] for K in [2,3,4]]:
        L.configure(task='count',word_len=n,alphabet=A,mod=K,device='cpu')
        params=L.handcoded_params(); rebuilt=independent.build('count',n,A,K)
        for mode in ['process','outcome']:
            for key,t in params[mode].items():
                assert torch.allclose(t,rebuilt[mode][key],atol=3e-5,rtol=3e-6),(n,A,K,mode,key)
        examples=[]
        for s0,q,word in itertools.product(range(K),range(A),itertools.product(range(A),repeat=n)):
            s=s0;states=[]
            for w in word:s=(s+(w==q))%K;states.append(s)
            examples.append((s0,q,word,states))
        minimum=100.;preds=0
        for mode in ['process','outcome']:
            params64={key:val.double() for key,val in params[mode].items()}
            for begin in range(0,len(examples),64):
                exs=examples[begin:begin+64]
                full=torch.tensor([L.row(ex,mode) for ex in exs])
                logits=L.forward(params64,full[:,:-1]);start=n+2
                selected=logits[:,start:];gold=full[:,start+1:]
                goldz=selected.gather(-1,gold.unsqueeze(-1)).squeeze(-1)
                wrong=selected.clone();wrong.scatter_(-1,gold.unsqueeze(-1),-torch.inf)
                margin=goldz-wrong.max(-1).values
                assert (margin>0).all(),(n,A,K,mode)
                minimum=min(minimum,margin.min().item());preds+=gold.numel()
                # Entire continuation, without teacher forcing.
                generated=full[:,:start+1].clone()
                for pos in range(start,full.shape[1]-1):
                    nxt=L.forward(params64,generated)[:,-1].argmax(-1)
                    assert torch.equal(nxt,full[:,pos+1]),(n,A,K,mode,pos)
                    generated=torch.cat([generated,nxt[:,None]],dim=1)
        rows.append(dict(n=n,alphabet=A,K=K,prompts=len(examples),verified_predictions=preds,min_margin=minimum))
        print(f'n={n} A={A} K={K}: {len(examples)} prompts, min margin {minimum:.8f}',flush=True)
(P/'evidence/executor_verification.json').write_text(json.dumps(rows,indent=2))
print('PASS: all',sum(r['prompts'] for r in rows),'prompts,',sum(r['verified_predictions'] for r in rows),'supervised predictions, both complete rollouts.')
