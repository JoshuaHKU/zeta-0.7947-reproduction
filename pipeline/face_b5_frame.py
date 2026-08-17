# -*- coding: utf-8 -*-
"""Face B5 -- the five-cycle frame: product identity and tracking.

Identity (proved in the paper, two lines): the 5-cycle lock
variance equals the pointwise product of the five lattice
autocorrelations, sum over the common shift t:
    sum_j N_5(j)^2 = sum_t A_1(t)...A_5(t).
Face: the product of per-lattice twin-singular-series models
(support t = 0 mod lcm) tracks the empirical five-fold product.
PASS: |emp/model - 1| <= 5% for every pattern.
"""
import math
import sys

import numpy as np

from common import lam_array, twin_singular_series, TWO_PI

PATTERNS = [(2, 3, 2, 3, 5), (3, 5, 3, 5, 2), (2, 3, 4, 3, 2),
            (5, 3, 5, 3, 3), (2, 5, 2, 5, 3), (3, 4, 3, 4, 5)]


def lcm_list(bs):
    out = 1
    for b in bs:
        out = out * b // math.gcd(out, b)
    return out


def face(T):
    X0 = int(T / TWO_PI)
    N = 1 << int(math.ceil(math.log2(16 * X0)))
    ok_all = True
    for bs in PATTERNS:
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
        L = lcm_list(bs)
        ts = np.arange(L, 3 * X0 + 1, L)
        emp = (float(np.prod([A[0] for A in As]))
               + 2 * float(np.sum(np.prod([A[ts] for A in As], axis=0))))
        mod = (float(np.prod([m[0] for m in models]))
               + 2 * float(np.sum(np.prod([m[ts] for m in models],
                                          axis=0))))
        track = abs(emp / mod - 1)
        good = track <= 0.05
        ok_all &= good
        print(f"  {bs}: tracking {track:.4f} "
              f"{'PASS' if good else 'FIRE'}")
    print(f"[b5 frame face] {'ALL PASS' if ok_all else 'FIRED'}")
    return ok_all


if __name__ == "__main__":
    face(float(sys.argv[1]) if len(sys.argv) > 1 else 19200.0)
