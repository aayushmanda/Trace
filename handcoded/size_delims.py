"""Size ( ) and [ ] around tall content in displayed math of Paper/*.tex: \\left( ... \\right).
Usage: python3 handcoded/size_delims.py [--dry] file.tex ..."""
import re, sys
TALL = ('\\frac', '\\sum', '\\prod', '\\int', '\\binom', '\\dfrac')
SIZED = re.compile(r'\\(left|bigl|Bigl|biggl|Biggl|big|Big|bigg|Bigg|bigr|Bigr|biggr|Biggr|right)\s*$')
DISP = re.compile(r'(\\\[)(.*?)(\\\])|(\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\})(.*?)(\\end\{\5\})', re.S)
def process(body):
    pairs, stack = [], []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '\\':
            i += 2; continue
        if ch in '([':
            stack.append((ch, i, bool(SIZED.search(body[:i]))))
        elif ch in ')]' and stack:
            op, j, sized = stack.pop()
            if (op, ch) in (('(', ')'), ('[', ']')) and not sized and not SIZED.search(body[:i]):
                inner = body[j + 1:i]
                if any(t in inner for t in TALL) and '&' not in inner and '\\\\' not in inner:
                    pairs.append((j, i))
        i += 1
    out = body
    for j, i in sorted(pairs, key=lambda p: -p[0]):
        out = out[:i] + '\\right' + out[i:]
        out = out[:j] + '\\left' + out[j:]
    return out, len(pairs)
def fix(text):
    total = [0]
    def sub(m):
        if m.group(1):
            new, k = process(m.group(2)); total[0] += k; return m.group(1) + new + m.group(3)
        new, k = process(m.group(6)); total[0] += k; return m.group(4) + new + m.group(7)
    return DISP.sub(sub, text), total[0]
if __name__ == '__main__':
    dry = '--dry' in sys.argv
    for f in [a for a in sys.argv[1:] if not a.startswith('--')]:
        s = open(f).read(); new, k = fix(s)
        print(f, k)
        if not dry: open(f, 'w').write(new)
