# -*- coding: utf-8 -*-
"""Face 1 -- the coincidence-counting local closure (Prop. 4.1).

For the b-cycle affine lock chain n_{t+1} = a_t*n_t + j_t mod p^s
the paper proves the pointwise identity N(j) = P*(p - D(j))/p with
D = #distinct{d_t mod p}, and the class closed form
kappa_b = (1 - E[D|class]/p) / (1 - 1/p)^b with E_j[kappa_b] = 1.

This module verifies all three statements by exhaustive
enumeration.  PASS criteria: zero pointwise violations; every
class matches the closed form to 1e-12; the weighted average
equals 1 to 1e-10.
"""
import itertools
import sys

import numpy as np


def cycle_check(b, p, s):
    """Exhaustive check at cycle length b, prime p, level p^s."""
    P = p ** s
    # Fixed unit coefficient family (skip multiples of p).
    a, x = [], 2
    while len(a) < b:
        x += 1
        if x % p:
            a.append(x % P)
    nfree = b - 1
    rest = ([y.ravel() for y in np.meshgrid(
        *([np.arange(P)] * (nfree - 1)), indexing='ij')]
        if nfree > 1 else [])
    n1 = np.arange(P)
    cnt, tot, cf_bad = {}, {}, 0
    for j1 in range(P):
        js = [np.full(P ** (nfree - 1) if nfree > 1 else 1, j1)]
        js += rest
        M = js[0].size
        # Affine recursion A_{t+1} = a_t A_t, B_{t+1} = a_t B_t + j_t.
        A, B = [1], [np.zeros(M, dtype=np.int64)]
        for t in range(1, b):
            A.append((A[-1] * a[t - 1]) % P)
            B.append((B[-1] * a[t - 1] + js[t - 1]) % P)
        # Brute-force count of n1 with all b points prime to p.
        NN = np.zeros(M, dtype=np.int64)
        for lo in range(0, M, 1 << 18):
            sl = slice(lo, min(lo + (1 << 18), M))
            acc = np.ones((P, NN[sl].size), dtype=bool)
            for t in range(b):
                pts = (A[t] * n1[:, None] + B[t][None, sl]) % P
                acc &= (pts % p) != 0
            NN[sl] = acc.sum(axis=0)
        # Pointwise identity: N == P*(p - D)/p with D from the
        # normalised offsets d_t = B_t / A_t mod p.
        ds = np.stack([(B[t] * pow(A[t], -1, P)) % p
                       for t in range(b)])
        D = np.array([len(set(ds[:, i].tolist())) for i in range(M)])
        cf_bad += int((NN != (P // p) * (p - D)).sum())
        # Aggregate by level-1 lock pattern (p | j_t or not).
        vpat = np.zeros(M, dtype=np.int64)
        for t in range(nfree):
            vpat = vpat * 2 + ((js[t] % p) == 0)
        for vp in np.unique(vpat):
            m = vpat == vp
            cnt[vp] = cnt.get(vp, 0) + int(NN[m].sum())
            tot[vp] = tot.get(vp, 0) + int(m.sum())
    # Closed form: kappa = (1 - E[D|pattern]/p)/(1-1/p)^b where
    # E[D|pattern] is computed by exact enumeration of unit locks.
    worst = 0.0
    for vp in sorted(cnt):
        kap = (cnt[vp] / tot[vp] / P) / (1 - 1 / p) ** b
        edges = [int(c) for c in format(vp, f"0{b-1}b")]
        free = [t for t, e in enumerate(edges) if e == 0]
        sD, cD = 0, 0
        for jv in itertools.product(range(1, p), repeat=len(free)):
            full = [0] * (b - 1)
            for idx, t in enumerate(free):
                full[t] = jv[idx]
            d = [0]
            for t in range(b - 1):
                d.append((d[-1] + full[t]) % p)
            sD += len(set(d))
            cD += 1
        pred = (1 - (sD / cD) / p) / (1 - 1 / p) ** b
        worst = max(worst, abs(kap / pred - 1))
    # Mean-one identity with the level-1 class measure.
    wtot = ktot = 0.0
    for vp in cnt:
        nz = bin(vp).count('1')
        w = (1 / p) ** nz * (1 - 1 / p) ** (b - 1 - nz)
        wtot += w
        ktot += w * (cnt[vp] / tot[vp] / P) / (1 - 1 / p) ** b
    mean_dev = abs(ktot / wtot - 1)
    ok = cf_bad == 0 and worst < 1e-12 and mean_dev < 1e-10
    print(f"  b={b} p={p} s={s}: pointwise violations {cf_bad}, "
          f"worst class dev {worst:.1e}, mean-1 dev {mean_dev:.1e} "
          f"{'PASS' if ok else 'FIRE'}")
    return ok


def main():
    print("[local closure] exhaustive verification:")
    ok = True
    for (b, p, s) in [(4, 5, 2), (5, 5, 2), (6, 5, 2),
                      (5, 7, 2), (6, 7, 1), (5, 11, 1), (6, 11, 1)]:
        ok &= cycle_check(b, p, s)
    print(f"[local closure] {'ALL PASS' if ok else 'FIRED'}")


if __name__ == "__main__":
    main()
