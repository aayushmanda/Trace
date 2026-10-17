from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parents[1];R=P.parent
# Audit all source routing pairs and cache compact per-seed measurements.
sourcefiles=['twoblock_basin_n6_a.json','twoblock_basin_n8_a.json','twoblock_basin_n12_a.json','twoblock_basin_n12_b.json','twoblock_n16.json']
rows=[]
for filename in sourcefiles:
 p=R/'results/step_reach'/filename
 for d in json.loads(p.read_text()):
  first=next((x['step'] for x in d['log'] if x['exact']==1.),None)
  op=R/f"results/step_reach_outcome/n{d['n']}_s{d['seed']}.json";od=json.loads(op.read_text())[0]
  rows.append(dict(n=d['n'],seed=d['seed'],step_solve_update=first,step_final=d['log'][-1]['exact'],outcome_final=od['log'][-1]['exact'],step_source=str(p.relative_to(R)),step_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),outcome_source=str(op.relative_to(R)),outcome_sha256=hashlib.sha256(op.read_bytes()).hexdigest()))
(P/'evidence/routing_rows.json').write_text(json.dumps(rows,indent=2))
print('Routing')
for n in [6,8,12,16]:
 rr=[r for r in rows if r['n']==n];t=[r['step_solve_update'] for r in rr if r['step_solve_update'] is not None];print(n,len(t),len(rr),np.median(t),t)
