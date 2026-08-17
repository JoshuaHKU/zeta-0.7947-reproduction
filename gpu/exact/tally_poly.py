# -*- coding: utf-8 -*-
"""汇总多面体法逐项结果，验全部门。"""
import glob, re
from fractions import Fraction as F

def tally(pat):
    tot, seen = F(0), {}
    for f in glob.glob(pat):
        for ln in open(f):
            m = re.search(r"term (\d+) sign (-?\d+) value (\S+)", ln)
            if m:
                seen[int(m.group(1))] = int(m.group(2)) * F(m.group(3))
    for v in seen.values():
        tot += v
    return tot, len(seen)

print("=== 多面体法 / polytope route ===")
g42 = F(0); ok42 = True
for D, tg, n in ((1, F(-2,315), 26), (2, F(-1,1260), 26), (3, F(-1,252), 26)):
    t, k = tally(f"poly/42{D}_*.txt")
    st = ("MATCH ✓" if t == tg else "MISMATCH ✗") if k == n else f"({k}/{n})"
    print(f"  U{D}: {k:2d}/{n}  {str(t):>16s} = {float(t):+.15f}  目标 {tg}  {st}")
    g42 += (6 if D < 3 else 3) * t
    if k != n: ok42 = False
if ok42:
    tg = F(-23,420)
    print(f"  ==> {{4,2}} = 6U1+6U2+3U3 = {g42} = {float(g42):+.15f}")
    print(f"      目标 {tg} = {float(tg):+.15f}   "
          f"{'MATCH ✓✓' if g42 == tg else 'MISMATCH ✗'}")
for b, tg, n in ((5, F(1,36), 150), (6, F(-1,126), 1082)):
    t, k = tally(f"poly/cycle{b}_*.txt")
    st = ("MATCH ✓✓" if t == tg else "MISMATCH ✗") if k == n else f"(进行中 {k}/{n})"
    nm = {5: "C5", 6: "{6}"}[b]
    print(f"  {nm}: {k}/{n}  {str(t):>16s} = {float(t):+.15f}  目标 {tg}  {st}")
