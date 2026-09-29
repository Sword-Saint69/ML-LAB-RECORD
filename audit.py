import re
t = open("opw_audit.txt", encoding="utf-8").read()
toks = re.split(r"Page No\. (\d+)", t)
pages = {}
prev = 1
buf = toks[0]
for k in range(1, len(toks), 2):
    n = int(toks[k])
    pages[n] = buf
    buf = toks[k + 1] if k + 1 < len(toks) else ""
nums = sorted(pages)
print("observed footers:", nums[0], "..", nums[-1], "count:", len(nums))
missing = [i for i in range(1, nums[-1] + 1) if i not in pages]
print("blank pages (no footer):", missing)
for n in nums:
    p = pages[n]
    heads = []
    for h in ["EXPERIMENT NO.", "PROGRAM", "OUTPUT", "RESULT", "INDEX"]:
        if h in p:
            heads.append(h)
    if heads:
        print(f"p{n:2d} ({'odd-RIGHT' if n % 2 else 'even-LEFT '}): {heads}")
