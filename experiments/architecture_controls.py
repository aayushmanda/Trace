"""Resumable oracle-architecture controls with a common validation-selected LR.

Calibration uses a separate initialization seed and the same rate grid for both
supervision modes. A single clipped LR is selected per architecture from pooled
validation performance. Confirmation uses five fresh seeds, with and without
clipping at that same selected rate. Test labels never enter model selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import torch
import compare_executor_rules as c
h=c.h
ROOT=Path(__file__).resolve().parents[1]


def json_write(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def loss_on(model,data,batch=128):
    total=0.
    with torch.no_grad():
        for start in range(0,len(data),batch):
            chunk=data.select(slice(start,start+batch))
            total+=h.language_model_loss(model,chunk).item()*len(chunk)
    return total/len(data)


def evaluate(model,circuits,data,tok,device,mode):
    model.eval()
    return {**c.generation_metrics(model,circuits,tok,device,mode),'cross_entropy':loss_on(model,data)}


def model_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def run_one(args):
    folder=args.output
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        print('already complete',folder,flush=True);return
    torch.set_num_threads(args.threads)
    device=torch.device(args.device); tok=h.make_tokenizer()
    train=c.unique_circuits(args.train_size,123,args.depth)
    val=c.unique_circuits(args.val_size,8000,args.depth,c.circuit_keys(train))
    # Calibration never constructs or evaluates the test split.
    data=h.encode_dataset(train,tok,args.mode).to(device)
    vdata=h.encode_dataset(val,tok,args.mode).to(device)
    builder={'process':h.build_random_trainable_process_architecture,
             'outcome':h.build_random_trainable_outcome_architecture}[args.architecture]
    model=builder(tok,args.depth,seed=args.seed,device=device)
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                        for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.)
    schedule=h.make_batch_schedule(len(train),args.steps,args.batch_size,2026)
    checks=sorted({0,args.steps}|{n for n in [100,500,1000,2000,4000,8000] if n<=args.steps})
    config={**vars(args),'output':str(folder),'initial_state_sha256':initial_hash,
            'parameters':sum(p.numel() for p in model.parameters()),
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),Path(c.__file__),Path(h.__file__)]},
            'data_seeds':{'train':123,'validation':8000,'test':9000,'batch':2026},
            'grid_selection':'one LR per architecture, average best validation accuracy of both formats; ties use NLL, then lower LR',
            'checkpoint_selection':'highest validation answer accuracy, then lowest validation NLL, then earliest checkpoint'}
    state_path=folder/'resume.pt'; records=[]; start=0; best=None; best_key=None; last_norms={}
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=True)
        for key in ['architecture','mode','seed','lr','clip','steps','train_size','val_size','depth','batch_size']:
            if saved['config'][key]!=config[key]: raise ValueError(f'Cannot resume changed {key}')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['step'];records=saved['records'];best=saved['best'];best_key=tuple(saved['best_key'])
    json_write(folder/'config.json',config)
    began=time.monotonic()
    def checkpoint(step):
        nonlocal best,best_key
        val_metrics=evaluate(model,val,vdata,tok,device,args.mode)
        if not math.isfinite(val_metrics['cross_entropy']):
            raise FloatingPointError('Non-finite validation loss')
        key=(val_metrics['free_answer_accuracy'],-val_metrics['cross_entropy'],-step)
        if best_key is None or key>best_key:
            best_key=key;best={'step':step,**val_metrics}
            torch.save({'model':model_state(model),'config':config,'selection':best},folder/'best.pt')
        records.append({'step':step,'validation':val_metrics,'gradient_norms_preclip':last_norms})
        torch.save({'model':model_state(model),'optimizer':optimizer.state_dict(),'config':config,
                    'step':step,'records':records,'best':best,'best_key':list(best_key)},state_path)
        json_write(folder/'trajectory.json',records)
        print(json.dumps({'architecture':args.architecture,'mode':args.mode,'seed':args.seed,
                          'lr':args.lr,'clip':args.clip,'step':step,'validation':val_metrics,
                          'elapsed_seconds':round(time.monotonic()-began,1)}),flush=True)
    try:
        if not records:checkpoint(0)
        for step in range(start+1,args.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            index=torch.tensor(schedule[step-1],device=device)
            loss=h.language_model_loss(model,data.select(index))
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip if args.clip>0 else float('inf'))
            if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient norm')
            if step in checks:
                # Block norms are computed after clipping; recover the preclip
                # norm with the same common scaling factor used by clip_grad_norm_.
                scale=min(1.,args.clip/(norm.item()+1e-6)) if args.clip>0 else 1.
                last_norms={'all':norm.item()}
                for i,block in enumerate(getattr(model,'blocks',[])):
                    last_norms[f'block_{i}']=math.sqrt(sum(p.grad.float().square().sum().item()
                                                         for p in block.parameters() if p.grad is not None))/scale
            optimizer.step()
            if step in checks:checkpoint(step)
        result={'status':'complete','config':config,'best_validation':best,
                'final_validation':records[-1]['validation'],'elapsed_seconds':time.monotonic()-began}
        if args.stage=='confirm':
            test=c.unique_circuits(args.test_size,9000,args.depth,c.circuit_keys(train+val))
            tdata=h.encode_dataset(test,tok,args.mode).to(device)
            result['final_test']=evaluate(model,test,tdata,tok,device,args.mode)
            result['final_train_sample']=evaluate(model,train[:300],data.select(slice(0,300)),tok,device,args.mode)
            selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(selected['model'])
            result['selected_test']=evaluate(model,test,tdata,tok,device,args.mode)
            accs=[r['validation']['free_answer_accuracy'] for r in records]
            result['persistent_validation_95_step']=next((records[i]['step'] for i,a in enumerate(accs)
                                                         if a>=.95 and all(x>=.95 for x in accs[i:])),None)
        json_write(folder/'result.json',result)
    except FloatingPointError as error:
        json_write(folder/'result.json',{'status':'unstable','config':config,'reason':str(error),
                                        'last_completed_checkpoint':records[-1]['step'] if records else None})
        print('unstable',error,flush=True)


def plan(args):
    protocol={'depth':4,'rates':[.002,.001,.0005,.0002,.0001],
              'calibration_seed':7001,'calibration_steps':args.calibration_steps,
              'confirmation_seeds':[42,43,44,45,46],'confirmation_steps':args.steps,
              'clips':[0.,1.],'train_size':20000,'val_size':1000,'test_size':1000,'batch_size':128,
              'weight_decay':0.,'init_std':.02,
              'selection':'Select one LR per architecture using clipped calibration, pooled over both formats. Repeat the chosen rate with clip=0 and clip=1 across all confirmation seeds.',
              'test_policy':'No test evaluation during calibration or checkpoint selection.',
              'calibration_runs':40,'confirmation_runs':40}
    path=args.output/'protocol.json';args.output.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol changed; use a new directory')
    json_write(path,protocol)
    return protocol


def orchestrate(args):
    protocol=plan(args)
    if args.action=='plan':
        print(json.dumps(protocol,indent=2));return
    jobs=[]
    if args.action=='calibrate':
        for arch in ['process','outcome']:
            for lr in protocol['rates']:
                for clip in protocol['clips']:
                    for mode in ['process','outcome']:
                        jobs.append((arch,mode,7001,lr,clip,'calibrate',args.calibration_steps))
    else:
        selected={}
        for arch in ['process','outcome']:
            scores=[]
            for lr in protocol['rates']:
                outcomes=[]
                for mode in ['process','outcome']:
                    f=args.output/'calibration'/f'{arch}_{mode}_s7001_lr{lr:g}_clip1'/'result.json'
                    if not f.exists():raise ValueError(f'Calibration incomplete: {f}')
                    outcomes.append(json.loads(f.read_text()))
                if all(x['status']=='complete' for x in outcomes):
                    scores.append((sum(x['best_validation']['free_answer_accuracy'] for x in outcomes)/2,
                                   -sum(x['best_validation']['cross_entropy'] for x in outcomes)/2,-lr))
            if not scores:raise RuntimeError(f'No stable calibration for {arch}')
            selected[arch]=-max(scores)[2]
        json_write(args.output/'selected_rates.json',selected)
        for arch,lr in selected.items():
            for seed in protocol['confirmation_seeds']:
                for clip in protocol['clips']:
                    for mode in ['process','outcome']:
                        jobs.append((arch,mode,seed,lr,clip,'confirm',args.steps))
    def worker(device, subset):
        for arch,mode,seed,lr,clip,stage,steps in subset:
            sub='calibration' if stage=='calibrate' else 'confirmation'
            folder=args.output/sub/f'{arch}_{mode}_s{seed}_lr{lr:g}_clip{clip:g}'
            folder.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,__file__,'single','--output',str(folder),'--architecture',arch,'--mode',mode,
                 '--seed',str(seed),'--lr',str(lr),'--clip',str(clip),'--stage',stage,'--steps',str(steps),
                 '--device',device]
            with (folder/'run.log').open('a') as f:
                run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            if run.returncode:raise RuntimeError(f'Failed: {folder}/run.log')
            print('completed',folder.name,flush=True)
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures=[pool.submit(worker,device,jobs[i::len(args.devices)]) for i,device in enumerate(args.devices)]
        for f in as_completed(futures):f.result()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','calibrate','confirm','single'])
    p.add_argument('--output',type=Path,default=ROOT/'results/architecture_controls')
    p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--calibration-steps',type=int,default=2000)
    p.add_argument('--devices',nargs='+',default=['cuda:2','cuda:3'])
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--architecture',choices=['process','outcome'],default='process')
    p.add_argument('--mode',choices=['process','outcome'],default='process')
    p.add_argument('--stage',choices=['calibrate','confirm'],default='calibrate')
    p.add_argument('--seed',type=int,default=7001);p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--clip',type=float,default=1.)
    p.add_argument('--depth',type=int,default=4)
    for name,value in [('train-size',20000),('val-size',1000),('test-size',1000),('batch-size',128)]:
        p.add_argument('--'+name,type=int,default=value)
    args=p.parse_args()
    if args.action=='single':run_one(args)
    else:orchestrate(args)


if __name__=='__main__':main()
