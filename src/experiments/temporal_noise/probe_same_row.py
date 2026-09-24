"""Post-hoc same-row prefix intervention on every completed checkpoint.

Keeps displayed first state and second operation identical. Changes the initial
state so that the same displayed first transition is correct versus coherent
wrong. The target successor at step two is therefore identical in each pair.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from handcoded.gates import phi
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.experiments.temporal_noise.run import gates_of, score_strings
from src.training.optim import build_gpt
from src.training.seed import configure_device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(4)
    device=configure_device(args.device,compile=False,bf16=False,distributed=False)
    root=Path('results/novelty_extension')
    paths=sorted((root/'canonical').glob('*/final.pt'))+sorted((root/'matched_patterns').glob('*/final.pt'))
    cache={}
    for index,file in enumerate(paths):
        if index%args.shards != args.shard:continue
        path=file.parent
        if (path/'same_row_prefix.json').exists():continue
        protocol=json.loads((path/'protocol.json').read_text())
        seed=protocol['train_seed']
        task=TASKS['boolean_circuit_8']
        if seed not in cache:
            train=generate_unique(task,20000,seed)
            instances=generate_unique(task,1000,101,{i.prompt for i in train})
            texts=[];spans=[];targets=[]
            for inst in instances:
                gates=gates_of(inst);start=int(inst.prompt[1:5],2)
                displayed=phi(start,gates[0])
                alternative=next(s for s in range(16) if phi(s,gates[0]) ^ _coherent_wrong_mask(gates[0]) == displayed)
                assert alternative != start
                assert phi(alternative,gates[0]) != displayed
                targets.append(phi(displayed,gates[1]))
                for source in [start,alternative]:
                    prompt=f'i{source:04b}'+inst.prompt[5:]
                    prefix=f'{prompt} {gates[0]}>{displayed:04b} {gates[1]}>'
                    for nxt in range(16):
                        texts.append(prefix+f'{nxt:04b}');spans.append([(len(prefix),len(prefix)+4)])
            cache[seed]=(texts,spans,np.asarray(targets))
        texts,spans,targets=cache[seed]
        model=build_gpt(task,SimpleNamespace(embedding=128,heads=4,layers=2,dropout=0.),device)
        checkpoint=torch.load(file,map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model']);model.eval()
        logs=score_strings(model,task.tokenizer,texts,spans,device,256).reshape(-1,2,16)
        probs=np.exp(logs)
        true=np.take_along_axis(probs,np.broadcast_to(targets[:,None,None],(len(targets),2,1)),axis=-1)[...,0]
        normalized=true/probs.sum(-1)
        record=dict(correct_prefix_probability=float(true[:,0].mean()),wrong_prefix_probability=float(true[:,1].mean()),
            same_row_gap=float((true[:,0]-true[:,1]).mean()),same_row_conditional_gap=float((normalized[:,0]-normalized[:,1]).mean()),
            n_prompts=len(targets),note='Post-hoc same displayed source and operation; initial prompt state changes. Not an extra training replicate.')
        np.savez_compressed(path/'same_row_predictions.npz',logp=logs,targets=targets)
        (path/'same_row_prefix.json').write_text(json.dumps(record,indent=2)+'\n')
        print(path,record['same_row_gap'],flush=True)
        del model,checkpoint
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
