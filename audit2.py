import re
t = open("full.txt", encoding="utf-8").read()
toks = re.split(r"Page No\. (\d+)", t)
pages = {}
prev_end = 0
prev_n = 0
for k in range(1, len(toks), 2):
    n = int(toks[k])
    pages[n] = toks[k - 1] if k - 1 >= 0 else ""
last = max(pages)
print("last footer:", last)
for n in sorted(pages):
    p = pages[n]
    heads = []
    for h in ["EXPERIMENT NO.", "PROGRAM", "OUTPUT", "RESULT", "INDEX"]:
        if h in p:
            heads.append(h)
    if heads:
        print(f"p{n:2d} ({'odd-RIGHT' if n % 2 else 'even-LEFT '}): {heads}")
