r"""Token-level rename of single-letter math symbols, math mode only, with recursion into \why{}/\text{}."""
import re, sys
FONT = r'\\(?:bm|mathbf|mathbb|mathrm|mathcal|mathsf|boldsymbol|operatorname|texttt|hat\{\\bm)'
VERB = re.compile(r'\\(cref|Cref|ref|label|eqref|citep|citet|cite|includegraphics|input)\{|\\(begin|end)\{(?!(?:equation|align|alignat|gather|multline)\*?\})')
TEXTARG = re.compile(r'\\(why|text|emph|textbf|mbox)\{')
MATHENV = r'equation|align|alignat|gather|multline'
def braced(s,j):
    d=1
    while j<len(s) and d:
        d += {'{':1,'}':-1}.get(s[j],0); j+=1
    return j
def run(s, sub, math=False):
    out=[]; i=0; n=len(s); disp=False
    while i<n:
        c=s[i]
        if s.startswith('\\$',i): out.append('\\$'); i+=2; continue
        m=VERB.match(s,i)
        if m: j=braced(s,m.end()); out.append(s[i:j]); i=j; continue
        m=TEXTARG.match(s,i)
        if m:
            j=braced(s,m.end()); out.append(s[i:m.end()]+run(s[m.end():j-1],sub,False)+'}'); i=j; continue
        m=re.match(FONT+r'(\s*\{|\s*[A-Za-z]|)', s[i:])
        if m and (math or disp):
            k=i+len(m.group(0))
            if m.group(1).strip().startswith('{'): k=braced(s,k)
            out.append(s[i:k]); i=k; continue
        if c=='$' and not disp: math=not math; out.append(c); i+=1; continue
        m=re.match(r'\\\[|\\begin\{(?:'+MATHENV+r')\*?\}', s[i:])
        if m and not math and not disp: disp=True; out.append(m.group(0)); i+=len(m.group(0)); continue
        m=re.match(r'\\\]|\\end\{(?:'+MATHENV+r')\*?\}', s[i:])
        if m and disp: disp=False; out.append(m.group(0)); i+=len(m.group(0)); continue
        if c=='\\':
            m=re.match(r'\\[A-Za-z]+|\\.', s[i:]); w=m.group(0)
            nxt=s[i+len(w)] if i+len(w)<n else ''
            rep=sub['cmd'].get(w,w) if (math or disp) else w
            if w=='\\Delta' and nxt=='_': rep=w
            if rep!=w and nxt.isalpha(): rep+=' '
            out.append(rep); i+=len(w); continue
        if (math or disp) and c.isalpha():
            prev=s[i-1] if i else ''
            if prev in '_^' : out.append(c); i+=1; continue   # bare subscript/superscript label like W_Q
            nxt=s[i+1] if i+1<n else ''
            rep=sub['let'].get(c)
            if c=='r' and nxt=='_': rep=sub.get('r')
            if rep is None: out.append(c)
            else: out.append(rep+(' ' if nxt.isalpha() else ''))
            i+=1; continue
        out.append(c); i+=1
    return ''.join(out)
def apply(path, lo, hi, sub):
    L=open(path).read().split('\n'); seg='\n'.join(L[lo-1:hi])
    for a,b in sub.get('pre',[]): seg=seg.replace(a,b)
    seg=run(seg,sub); L[lo-1:hi]=seg.split('\n'); open(path,'w').write('\n'.join(L))
