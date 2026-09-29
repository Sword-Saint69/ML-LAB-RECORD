import gzip
import re

raw = gzip.open("OPWER.synctex.gz", "rt", encoding="utf-8", errors="replace").read().split("\n")
in_main = False
depth = 0
pages = {}
for ln in raw:
    if ln.startswith("{1"):
        in_main = True
        depth += 1
        continue
    if in_main and re.match(r"^\}\d+$", ln):
        depth -= 1
        if depth <= 0:
            in_main = False
        continue
    if in_main and ln.startswith("x"):
        m = re.match(r"x(\d+),\d+:[\d.\-]+,[\d.\-]+:(\d+),", ln)
        if m:
            line = int(m.group(1))
            if line not in pages:
                pages[line] = int(m.group(2))

targets = [125, 185, 216, 234, 251, 307, 344, 401, 418, 475, 514, 547, 581, 598,
           651, 701, 722, 739, 792, 842, 860, 877, 933, 983, 1003, 1020, 1084,
           1129, 1151, 1172, 1189, 1242, 1284, 1304, 1321, 1377, 1421, 1447,
           1464, 1518, 1564, 1586, 1611]
labels = ["1A-sheet", "1A-prog", "1A-out", "1A-res", "1B-sheet", "1B-prog", "1B-out", "1B-res",
          "2-sheet", "2-prog", "2-progC", "2-out", "2-res", "3A-sheet", "3A-prog", "3A-out", "3A-res",
          "3B-sheet", "3B-prog", "3B-out", "3B-res", "4A-sheet", "4A-prog", "4A-out", "4A-res",
          "4B-sheet", "4B-prog", "4B-progC", "4B-out", "4B-res", "4C-sheet", "4C-prog", "4C-out", "4C-res",
          "5-sheet", "5-prog", "5-out", "5-res", "6-sheet", "6-prog", "6-progC", "6-out", "6-res"]
for lab, ln in zip(labels, targets):
    print(f"{lab:10s} tex:{ln} -> p{pages.get(ln, '??')}")
