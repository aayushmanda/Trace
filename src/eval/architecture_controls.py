"""Resumable oracle-architecture controls with a common validation-selected LR.

Semantic-token / handcoded stack. Report the fraction of confirmation seeds
with test accuracy > 95% after validation-only LR selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from src.training.config import load_yaml
from src.training.io import write_csv
from src.training.optim import make_adamw
from src.training.progress import progress
from src.training.seed import add_compile_bf16_flags, autocast_context, configure_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval import executor_comparison as c
h = c.h


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
    set_seed(args.seed)
    device=torch.device(args.device)
    args.compile = False  # run_one is always eager; do not lie in config.json
    configure_device(device, compile=False, bf16=getattr(args, "bf16", None))
    tok=h.make_tokenizer()
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
    optimizer=make_adamw(model.parameters(), args.lr, weight_decay=0., device=device)
    schedule=h.make_batch_schedule(len(train),args.steps,args.batch_size,2026)
    checks=sorted({0,args.steps}|{n for n in [100,500,1000,2000,4000,8000] if n<=args.steps})
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    config.update({'output':str(folder),'initial_state_sha256':initial_hash,
            'parameters':sum(p.numel() for p in model.parameters()),
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),Path(c.__file__),Path(h.__file__)]},
            'data_seeds':{'train':123,'validation':8000,'test':9000,'batch':2026},
            'grid_selection':'one LR per architecture, average best validation accuracy of both formats; ties use NLL, then lower LR',
            'checkpoint_selection':'highest validation answer accuracy, then lowest validation NLL, then earliest checkpoint'})
    state_path=folder/'resume.pt'; records=[]; start=0; best=None; best_key=None; last_norms={}
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
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
        for step in progress(range(start+1,args.steps+1), desc=f"{args.architecture}/{args.mode}/s{args.seed}", leave=False):
            model.train();optimizer.zero_grad(set_to_none=True)
            index=torch.tensor(schedule[step-1],device=device)
            with autocast_context(device):
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
            selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
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
    protocol={'depth':args.depth,'rates':list(args.rates),
              'calibration_seed':7001,'calibration_steps':args.calibration_steps,
              'confirmation_seeds':list(args.confirmation_seeds),'confirmation_steps':args.steps,
              'clips':[0.,1.],'train_size':args.train_size,'val_size':args.val_size,
              'test_size':args.test_size,'batch_size':args.batch_size,
              'weight_decay':0.,'init_std':.02,
              'selection':'Select one LR per architecture using clipped calibration, pooled over both formats. Repeat the chosen rate with clip=0 and clip=1 across all confirmation seeds.',
              'test_policy':'No test evaluation during calibration or checkpoint selection.',
              'success_threshold':0.95,
              'calibration_runs':2*len(args.rates)*2*2,
              'confirmation_runs':2*len(args.confirmation_seeds)*2*2}
    path=args.output/'protocol.json';args.output.mkdir(parents=True,exist_ok=True)
    if path.exists():
        print('keeping existing', path, flush=True)
        return json.loads(path.read_text())
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
            cmd=[sys.executable,'-m','src','architecture','single','--output',str(folder),'--architecture',arch,'--mode',mode,
                 '--seed',str(seed),'--lr',str(lr),'--clip',str(clip),'--stage',stage,'--steps',str(steps),
                 '--device',device,'--depth',str(protocol['depth']),
                 '--train-size',str(protocol['train_size']),'--val-size',str(protocol['val_size']),
                 '--test-size',str(protocol['test_size']),'--batch-size',str(protocol['batch_size']),
                 '--threads',str(args.threads)]
            with (folder/'run.log').open('a') as f:
                run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            if run.returncode:raise RuntimeError(f'Failed: {folder}/run.log')
            print('completed',folder.name,flush=True)
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures=[pool.submit(worker,device,jobs[i::len(args.devices)]) for i,device in enumerate(args.devices)]
        for f in as_completed(futures):f.result()


def summarize(args):
    """Fraction of confirmation seeds with test answer accuracy above 95%."""
    folder=args.output/'confirmation'
    if not folder.exists():
        raise FileNotFoundError(folder)
    rows=[]
    for path in sorted(folder.glob('*/result.json')):
        rec=json.loads(path.read_text())
        cfg=rec.get('config',{})
        test=rec.get('selected_test') or rec.get('final_test') or {}
        acc=test.get('free_answer_accuracy')
        rows.append({
            'architecture':cfg.get('architecture'),
            'mode':cfg.get('mode'),
            'seed':cfg.get('seed'),
            'lr':cfg.get('lr'),
            'clip':cfg.get('clip'),
            'status':rec.get('status'),
            'test_accuracy':acc,
            'success':None if acc is None else acc>=0.95,
        })
    out=args.output/'success_fraction.csv'
    if rows:
        write_csv(out, rows)
    summary=[]
    by={}
    for row in rows:
        key=(row['architecture'],row['mode'],row['lr'],row['clip'])
        by.setdefault(key,[]).append(row)
    print('architecture mode lr clip n n_ok fraction_gt_95')
    for key,group in sorted(by.items()):
        done=[r for r in group if r['status']=='complete' and r['success'] is not None]
        n_ok=sum(r['success'] for r in done)
        frac=n_ok/len(done) if done else float('nan')
        print(*key,len(done),n_ok,f'{frac:.2f}' if done else 'na')
        summary.append({'architecture':key[0],'mode':key[1],'lr':key[2],'clip':key[3],
                        'n':len(done),'n_success':n_ok,'fraction_gt_95':frac})
    json_write(args.output/'success_fraction.json',summary)
    print('wrote',out)


def main():
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config',default='configs/experiments/architecture_controls.yaml')
    pre_args,_=pre.parse_known_args()
    cfg=load_yaml(pre_args.config) if Path(pre_args.config).exists() or (ROOT/pre_args.config).exists() else {}
    p=argparse.ArgumentParser(description=__doc__, parents=[pre])
    p.add_argument('action',choices=['plan','calibrate','confirm','single','summarize'])
    p.add_argument('--output',type=Path,default=ROOT/str(cfg.get('output','results/architecture_controls_n10')))
    p.add_argument('--steps',type=int,default=int(cfg.get('steps',8000)))
    p.add_argument('--calibration-steps',type=int,default=int(cfg.get('calibration_steps',2000)))
    p.add_argument('--devices',nargs='+',default=list(cfg.get('devices',['cuda:2','cuda:3'])))
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--architecture',choices=['process','outcome'],default='process')
    p.add_argument('--mode',choices=['process','outcome'],default='process')
    p.add_argument('--stage',choices=['calibrate','confirm'],default='calibrate')
    p.add_argument('--seed',type=int,default=7001);p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--clip',type=float,default=1.)
    p.add_argument('--depth',type=int,default=int(cfg.get('depth',4)))
    for name,value in [('train-size',int(cfg.get('train_size',20000))),
                       ('val-size',int(cfg.get('val_size',1000))),
                       ('test-size',int(cfg.get('test_size',1000))),
                       ('batch-size',int(cfg.get('batch_size',128)))]:
        p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--rates',nargs='+',type=float,default=list(cfg.get('rates',[.002,.001,.0005,.0002])))
    p.add_argument('--confirmation-seeds',nargs='+',type=int,
                   default=list(cfg.get('confirmation_seeds',[42,43,44,45,46,47,48,49,50,51])))
    add_compile_bf16_flags(p, cfg)
    args=p.parse_args()
    if args.action=='single':run_one(args)
    elif args.action=='summarize':summarize(args)
    else:orchestrate(args)


if __name__=='__main__':main()
