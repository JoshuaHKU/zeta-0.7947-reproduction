# -*- coding: utf-8 -*-
"""Face B6 -- the six-cycle frame: Bell(6) gate and product face.

(a) Bell(6) = 203 class table, hexagon chords 30/15, matching
    crossing spectrum 5/6/3/1 (Touchard-Riordan).
(b) Six-fold product identity: the 6-cycle lock variance equals
    sum_t A_1(t)...A_6(t); the twin-singular-series product model
    tracks it.  PASS: exact counts; tracking <= 5% per pattern.
"""
import itertools
import math
import sys

import numpy as np

from common import lam_array, twin_singular_series, TWO_PI

PATTERNS = [(2, 3, 2, 3, 2, 3), (2, 3, 2, 5, 2, 5),
            (3, 5, 3, 5, 3, 5), (2, 3, 4, 3, 2, 3),
            (5, 3, 5, 3, 5, 3), (2, 5, 3, 5, 2, 3)]


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def crossing(a, b):
    a1, a2 = sorted(a)
    b1, b2 = sorted(b)
    return (a1 < b1 < a2 < b2) or (b1 < a1 < b2 < a2)


def gate():
    parts = list(set_partitions(list(range(6))))
    counts, c2211, mc = {}, {'a': 0, 'o': 0}, {}
    for p in parts:
        sig = tuple(sorted((len(b) for b in p), reverse=True))
        counts[sig] = counts.get(sig, 0) + 1
        pr = [b for b in p if len(b) == 2]
        if sig == (2, 2, 1, 1):
            c2211['o' if crossing(pr[0], pr[1]) else 'a'] += 1
        if sig == (2, 2, 2):
            nc = sum(1 for x, y in itertools.combinations(pr, 2)
                     if crossing(x, y))
            mc[nc] = mc.get(nc, 0) + 1
    expect = {(1,) * 6: 1, (2, 1, 1, 1, 1): 15, (2, 2, 1, 1): 45,
              (2, 2, 2): 15, (3, 1, 1, 1): 20, (3, 2, 1): 60,
              (3, 3): 10, (4, 1, 1): 15, (4, 2): 15, (5, 1): 6,
              (6,): 1}
    ok = (len(parts) == 203 and counts == expect
          and (c2211['a'], c2211['o']) == (30, 15)
          and mc == {0: 5, 1: 6, 2: 3, 3: 1})
    print(f"[o6 gate] Bell(6) = {len(parts)}, chords "
          f"{c2211['a']}/{c2211['o']}, matchings "
          f"{sorted(mc.items())} {'PASS' if ok else 'FIRE'}")
    return ok


def face(T):
    X0 = int(T / TWO_PI)
    N = 1 << int(math.ceil(math.log2(16 * X0)))
    ok_all = gate()
    for bs in PATTERNS:
        L = 1
        for b in bs:
            L = L * b // math.gcd(L, b)
        As, models = [], []
        for b in bs:
            n_lo, n_hi = X0 // b + 1, (4 * X0) // b
            lam = lam_array(n_hi + 10)
            x = np.zeros(N)
            nn = np.arange(n_lo, n_hi + 1)
            x[b * nn] = lam[nn]
            As.append(np.fft.irfft(np.abs(np.fft.rfft(x)) ** 2, n=N))
            K = n_hi - n_lo
            s = twin_singular_series(K + 10)
            model = np.zeros(N)
            ts = np.arange(0, N, b)
            u = ts // b
            sel = u <= K
            model[ts[sel]] = s[u[sel]] * np.maximum(K + 1 - u[sel], 0)
            model[0] = float((lam[nn] ** 2).sum())
            models.append(model)
        ts = np.arange(L, 3 * X0 + 1, L)
        emp = (float(np.prod([A[0] for A in As]))
               + 2 * float(np.sum(np.prod([A[ts] for A in As],
                                          axis=0))))
        mod = (float(np.prod([m[0] for m in models]))
               + 2 * float(np.sum(np.prod([m[ts] for m in models],
                                          axis=0))))
        track = abs(emp / mod - 1)
        good = track <= 0.05
        ok_all &= good
        print(f"  {bs}: tracking {track:.4f} "
              f"{'PASS' if good else 'FIRE'}")
    print(f"[b6 frame face] {'ALL PASS' if ok_all else 'FIRED'}")
    return ok_all


if __name__ == "__main__":
    face(float(sys.argv[1]) if len(sys.argv) > 1 else 19200.0)
