# -*- coding: utf-8 -*-
"""Direct Ursell cumulant engine for the sine-kernel determinantal
process: overlap volumes and connected correlators C_b, used as the
reference implementation by the F-CYC/F-TRACE gates (paper §5.5).
正弦核行列式过程的直接 Ursell 累积量引擎: 重叠体积与连通相关子 C_b,
作为 F-CYC/F-TRACE 门的参照实现 (论文 §5.5).
"""
import numpy as np, itertools, math, time
t0=time.time()

def ordered_set_partitions(slots):
    # all ordered tuples of disjoint nonempty sets covering slots
    def rec(rem):
        if not rem: yield []
        else:
            first = rem[0]
            rest = rem[1:]
            for r in range(len(rest)+1):
                for comb in itertools.combinations(rest, r):
                    grp = (first,)+comb
                    rem2 = tuple(x for x in rest if x not in comb)
                    for tail in rec(rem2):
                        yield [grp]+tail
    base = list(rec(tuple(slots)))
    out = []
    for part in base:                      # interleave orders of blocks
        for perm in itertools.permutations(part):
            out.append(list(perm))
    # dedupe (rec already fixes first element in first block; permuting gives all ordered)
    seen=set(); ded=[]
    for p in out:
        key=tuple(tuple(sorted(g)) for g in p)+("|",)+tuple(len(g) for g in p)
        k2=tuple(tuple(sorted(g)) for g in p)
        kk=(k2, tuple(map(tuple,p)))
        s=tuple(map(tuple,p))
        if s not in seen: seen.add(s); ded.append(p)
    return ded

def overlap(Qs):
    # length of {t: t+Q_j in [-1/2,1/2] all j} = (1 - (maxQ-minQ))_+
    mx = np.maximum.reduce(Qs); mn = np.minimum.reduce(Qs)
    return np.clip(1.0-(mx-mn), 0.0, None)

def C_of(vlist):
    # free-fermion cumulant C_b for Z-fields at frequencies vlist (arrays), sum(v)=0
    b = len(vlist)
    tot = 0.0
    for part in ordered_set_partitions(range(b)):
        m = len(part)
        coeff = (-1.0)**(m-1)/m
        gs = [sum(vlist[i] for i in g) for g in part]
        Qs = [np.zeros_like(vlist[0])]
        acc = np.zeros_like(vlist[0])
        for j in range(m-1):
            acc = acc + gs[j]; Qs.append(acc.copy())
        tot = tot + coeff*overlap(Qs)
    return tot

# gate b=2: C2(v) should equal min(|v|,1)
v = np.linspace(-2,2,801)
C2 = C_of([v, -v])
err2 = np.max(np.abs(C2 - np.minimum(np.abs(v),1.0)))
print(f"gate b=2: max|C2 - min(|v|,1)| = {err2:.2e}")

# gate b=3: assembly of m3 -> integral of O3*C3 should be ~0 (m3 = 1 + 1 + conn = 2)
dv=0.004; g1 = np.arange(-2,2+dv/2,dv)
V1,V2 = np.meshgrid(g1,g1,indexing='ij'); V3 = -V1-V2
O3 = overlap([np.zeros_like(V1), V1, V1+V2])
C3 = C_of([V1,V2,V3])
I3 = np.sum(O3*C3)*dv*dv
print(f"gate b=3: integral O3*C3 = {I3:+.5f}  (target 0; m3 = 2 = 1+1+{I3:+.4f})")
print(f"engine time {time.time()-t0:.0f}s")
