"""Run the predeclared D=2,4,6 semantic-token replication, with saved checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/executor_comparison/depth_replication')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--depths',nargs='+',type=int,default=[2,4,6])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    runner=ROOT/'experiments/compare_executor_rules.py'
    protocol={'depths':args.depths,'seeds':args.seeds,'steps':args.steps,
              'train_size':10000,'test_size':256,'probe_size':64,'batch_size':128,
              'd_model':96,'d_ff':192,'lr':0.002,'weight_decay':0.01,'clip':1,
              'device':'cpu','threads_per_run':1,'backgrounds':2,
              'checkpoints':[0,100,500,1000,args.steps],
              'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),
              'purpose':'depth replication of local readout and bridge test, not oracle-architecture controls'}
    mf=args.output/'protocol.json'
    if mf.exists() and json.loads(mf.read_text())!=protocol:
        raise ValueError('Protocol changed: select a new output directory')
    mf.write_text(json.dumps(protocol,indent=2)+'\n')
    def run_one(depth,seed):
        directory=args.output/f'depth{depth}_seed{seed}'
        manifest=directory/'manifest.json'
        if manifest.exists() and 'elapsed_seconds' in json.loads(manifest.read_text()):
            return depth,seed,'completed previously'
        if directory.exists():
            raise RuntimeError(f'Incomplete run exists: {directory}; inspect before retrying')
        command=[sys.executable,str(runner),'--output',str(directory),'--depth',str(depth),
                 '--seed',str(seed),'--steps',str(args.steps),'--device','cpu','--threads','1']
        log=args.output/f'depth{depth}_seed{seed}.log'
        with log.open('w') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Run failed: {log}')
        return depth,seed,'complete'
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_one,d,s) for d in args.depths for s in args.seeds]
        for future in as_completed(jobs):
            print(future.result(),'elapsed',round(time.monotonic()-started),flush=True)
    rows=[]
    for depth in args.depths:
        for seed in args.seeds:
            with (args.output/f'depth{depth}_seed{seed}'/'metrics.csv').open() as f:
                rows.extend({'depth':depth,**r} for r in csv.DictReader(f))
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    print('All runs and aggregate complete',args.output,flush=True)


if __name__=='__main__': main()
