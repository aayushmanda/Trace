r"""Apply a regex substitution only inside math mode, on lines lo..hi, skipping \why/\text/\mathrm/\cref/\label arguments."""
import re, sys
MATHENV = r'equation|align|alignat|gather|multline'
SKIP = re.compile(r'\\(why|text|mathrm|cref|Cref|ref|label|emph|textbf)\{')
def sub_math(s, pat, repl):
    out=[]; i=0; n=len(s); math=False; disp=None
    while i<n:
        if s.startswith('\\$',i): out.append('\\$'); i+=2; continue
        m = SKIP.match(s,i)
        if m:  # copy balanced braces verbatim
            j=m.end(); d=1
            while j<n and d: d += {'{':1,'}':-1}.get(s[j],0); j+=1
            out.append(s[i:j]); i=j; continue
        if not disp and s[i]=='$': math=not math; out.append('$'); i+=1; continue
        m = re.match(r'\\\[|\\begin\{('+MATHENV+r')\*?\}', s[i:])
        if m and not math and not disp: disp='x'; out.append(m.group(0)); i+=len(m.group(0)); continue
        m = re.match(r'\\\]|\\end\{('+MATHENV+r')\*?\}', s[i:])
        if m and disp: disp=None; out.append(m.group(0)); i+=len(m.group(0)); continue
        # take a run of plain chars
        if s[i]=='\\':
            m=re.match(r'\\[A-Za-z]+|\\.', s[i:]); j=i+len(m.group(0))
        else:
            j=i+1
            while j<n and s[j] not in '$\\': j+=1
        chunk=s[i:j]
        if (math or disp) and not chunk.startswith('\\'): chunk=re.sub(pat,repl,chunk)
        out.append(chunk); i=j
    return ''.join(out)
def apply(path, lo, hi, pat, repl):
    L=open(path).read().split('\n')
    seg='\n'.join(L[lo-1:hi]); seg=sub_math(seg,pat,repl)
    L[lo-1:hi]=seg.split('\n'); open(path,'w').write('\n'.join(L))
if __name__=='__main__':
    apply(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
