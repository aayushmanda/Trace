"""Static acceptance checks after a successful build."""
from pathlib import Path
import re,json,subprocess
P=Path(__file__).resolve().parents[1]
# Only included section files; .tex references are checked including unused macros.
text='\n'.join(p.read_text() for p in P.rglob('*.tex'))
labels=re.findall(r'\\label\{([^}]+)\}',text)
refs=[]
for m in re.finditer(r'\\(?:[cC]ref|ref|eqref)\{([^}]+)\}',text):refs.extend(m.group(1).split(','))
assert not set(refs)-set(labels),set(refs)-set(labels)
assert len(labels)==len(set(labels))
assert text.count('\\begin{theorem}')==1
assert text.count('\\begin{proposition}[Exact Outcome')==1
assert not re.search(r'undefined|multiply defined|Overfull', (P/'main.log').read_text())
assert not re.search(r'\\input\{[^}]*(two_path|matched_proof|clean_routing|full_prefix_routing)',text)
info=subprocess.check_output(['pdfinfo',str(P/'main.pdf')],text=True)
pagecount=int(re.search(r'Pages:\s*(\d+)',info).group(1));assert 28<=pagecount<=32
execrows=json.loads((P/'evidence/executor_verification.json').read_text())
assert len(execrows)==30 and min(x['min_margin'] for x in execrows)>19
assert '0 failure(s)' in (P/'evidence/theorem4_checks.log').read_text()
for name in ['main.tex','main.pdf','proof_audit.md','claims_matrix.md','CHANGELOG.md','Makefile']:
 assert (P/name).is_file()
summary={'pages':pagecount,'principal_learning_theorems':1,'main_executor_propositions':1,'undefined_references':0,'undefined_citations':0,'overfull_boxes':0,'enumerated_prompts':sum(x['prompts'] for x in execrows),'verified_predictions':sum(x['verified_predictions'] for x in execrows),'minimum_executor_margin':min(x['min_margin'] for x in execrows),'original_gradient_check_failures':0,'training_rerun':False}
(P/'evidence/acceptance.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
