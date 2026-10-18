"""Resolve numbered citations: paper_src.tex (\\cite{key}, \\rrow{key}) -> paper.tex with literal [n].  Numbering follows the order of rows in the reference table."""
import re, sys
d = "docs/paper/"; s = open(d + "paper_src.tex").read()
keys = re.findall(r"\\rrow\{([a-z0-9]+)\}", s); num = {k: i + 1 for i, k in enumerate(keys)}
def cite(m):
    ks = [k.strip() for k in m.group(1).split(",")]; miss = [k for k in ks if k not in num]
    if miss: sys.exit(f"unknown citation keys: {miss}")
    return "[" + ", ".join(str(num[k]) for k in ks) + "]"
body = re.sub(r"\\cite\{([^}]*)\}", cite, s); body = re.sub(r"\\rrow\{([a-z0-9]+)\}", lambda m: "\\textbf{[%d]}" % num[m.group(1)], body)
open(d + "paper.tex", "w").write(body)
cited = set(k for m in re.findall(r"\\cite\{([^}]*)\}", s) for k in m.split(","))
print(len(keys), "references;", "uncited:", [k for k in keys if k not in cited])
