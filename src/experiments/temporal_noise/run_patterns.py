"""Equal whole-trace quality control using blocked/interleaved latent flags.

Reuses the primary trainer with a replacement corpus factory. The legacy k0/k1
run tag encodes blocked/interleaved here; it is NOT a Markov persistence value.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from handcoded.gates import make_gate_names, phi
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.experiments.temporal_noise import run as base
from src.training.optim import build_gpt


def pattern_corpus(instances,rho,code,seed):
    flags=np.random.default_rng(seed).random((len(instances),2)) < rho
    clean=np.repeat(flags,4,axis=1) if code == 0 else np.tile(flags,(1,4))
    names=make_gate_names(); ids={g:i for i,g in enumerate(names)}
    counts=np.zeros((52,16,16),dtype=np.int64)
    states=np.empty((len(instances),8),dtype=np.int64); traces=[]
    for i,inst in enumerate(instances):
        state=int(inst.prompt[1:5],2); steps=[]
        assert len(base.gates_of(inst))==8
        for t,gate in enumerate(base.gates_of(inst)):
            truth=phi(state,gate)
            nxt=truth if clean[i,t] else truth ^ _coherent_wrong_mask(gate)
            counts[ids[gate],state,nxt]+=1
            assert (nxt==truth)==clean[i,t]
            states[i,t]=nxt
            steps.append(f'{gate}>{nxt:04b}');state=nxt
        trace=' '.join(steps)
        assert (trace==inst.correct_trace)==bool(clean[i].all())
        traces.append(trace)
    return traces,states,clean,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--code',type=int,choices=[0,1],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[2001,2002,2003])
    args=p.parse_args()
    args.depth=8;args.train_size=20000;args.eval_size=1000;args.steps=8000
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    base.make_corpus=pattern_corpus
    pattern='blocked' if args.code==0 else 'interleaved'
    for seed in args.seeds:
        tag=f's{seed}_rho0.3_k{args.code}_answer0'
        path=args.output/tag
        if (path/'pattern_metrics.json').exists():
            continue
        base.run_condition(args,seed,(.3,float(args.code),False))
        protocol=json.loads((path/'protocol.json').read_text())
        protocol.update(pattern=pattern,legacy_tag_code=args.code,
            corpus_factory_sha256=base.sha(Path(__file__).read_bytes()),
            theoretical_whole_trace_correctness=.09,
            theoretical_gold_prefix_probabilities=[.3,1,1,1,.3,1,1,1] if args.code==0 else [.3,.3,1,1,1,1,1,1])
        for key in ['persistence','theoretical_clean_after_clean','theoretical_clean_after_wrong']:
            protocol.pop(key,None)
        (path/'pattern_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        task=TASKS['boolean_circuit_8']
        train=generate_unique(task,20000,501+seed-2001)
        evaluation=generate_unique(task,1000,101,{i.prompt for i in train})
        shortened=[replace(i,prompt=i.prompt+' '+' '.join(i.correct_trace.split()[:2]),
                    correct_trace=' '.join(i.correct_trace.split()[2:])) for i in evaluation]
        model=build_gpt(task,SimpleNamespace(embedding=128,heads=4,layers=2,dropout=0.),torch.device(args.device))
        checkpoint=torch.load(path/'final.pt',map_location=args.device,weights_only=False)
        model.load_state_dict(checkpoint['model']);model.eval()
        metrics,records=base.rollout(model,task,shortened,args.device,False)
        (path/'anchor_two_generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        result=json.loads((path/'metrics.json').read_text())
        result.pop('persistence');result['pattern']=pattern
        result.update({f'anchor_two_{k}':v for k,v in metrics.items()})
        (path/'pattern_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        del model,checkpoint
        torch.cuda.empty_cache()
        print('PATTERN COMPLETE',tag,flush=True)


if __name__=='__main__':
    main()
