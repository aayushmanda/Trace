import re, sys
def fix(path, lo, hi):
    L = open(path).read().split('\n')
    for i in range(lo-1, hi):
        x = L[i]
        x = re.sub(r';s\)', r';\\cbg)', x)
        x = re.sub(r',s\)', r',\\cbg)', x)           # F_K(m,s), (m_b,s)
        x = x.replace('e^{-s-m', 'e^{-\\cbg-m')
        x = x.replace('(c,g,s,a)', '(c,g,\\cbg,a)').replace('(-c,-g,s,-a)', '(-c,-g,\\cbg,-a)')
        x = x.replace('\\ddt{s}', '\\ddt{\\cbg}').replace('\\dot s', '\\dot c_{\\mathrm{bg}}').replace('{\\mathcal L}{s}', '{\\mathcal L}{\\cbg}')
        x = x.replace('$m,s\\in', '$m,\\cbg\\in')
        x = re.sub(r'(?<![\\\w\'{])s(?=&=|=\\initscale\\bigl|=0\b|\(0\)=0|\+\\t?frac|-\\t?frac)', r'\\cbg', x)
        x = x.replace('$s+\\frac{m', '$\\cbg+\\frac{m').replace('$s-\\frac{m', '$\\cbg-\\frac{m')
        x = x.replace('contrast $s$', 'contrast $\\cbg$').replace('term and $s$ are', 'term and $\\cbg$ are')
        x = x.replace('omit $s$', 'omit $\\cbg$').replace('the $s$ coordinate', 'the $\\cbg$ coordinate')
        x = x.replace('no $s$ term', 'no $\\cbg$ term').replace('$c,g,s$ grow', '$c,g,\\cbg$ grow')
        x = x.replace('$K\\ge3$, $s$.', '$K\\ge3$, $\\cbg$.')
        L[i] = x
    open(path, 'w').write('\n'.join(L))
fix('body.tex', 240, 310)
fix('appendix.tex', 855, 1260)
