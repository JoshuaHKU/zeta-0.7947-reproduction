# -*- coding: utf-8 -*-
"""Face 2 -- the per-bin Siegel-Walfisz face (Thm 5.1's engine).

At a doubly-major FFT bin the single prime sum obeys
S(a/Q + eta) = mu(Q)/phi(Q) * I(eta) + E with |E| small (SW +
partial summation); the face measures the absolutely-normalised
error RMS(|S|^2 - |main|^2)/Y^2 over m-side major bins (Q <= 12,
squarefree) and reports its height trend.  Expected: steady decay
with no floor (the archive records a tenfold drop over a 64x
height range).  PASS: value at the requested height below the
registered envelope 6e-3.
"""
import math
import sys

import numpy as np

from common import build_spectra, arc_quality, mu_int, phi_int

FAMS = [(5, 3), (3, 2), (7, 2), (4, 3), (5, 2), (7, 4)]
P_CUT = 40


def face(T):
    rels = []
    for (b1, b2) in FAMS:
        am2, _, geo = build_spectra(T, b1, b2)
        N = geo['N']
        m_lo, m_hi = geo['m_span']
        L = m_hi - m_lo + 1
        wid_m = P_CUT / geo['Ym']
        errs = []
        for j in range(1, len(am2) - 1):
            gm = (b2 * j / N) % 1.0
            Q = arc_quality(gm, P_CUT, wid_m)
            if not Q or Q > 12 or mu_int(Q) == 0:
                continue
            a = round(gm * Q)
            if Q > 1 and math.gcd(a, Q) != 1:
                continue
            eta = gm - a / Q
            # Window main term: mu/phi times the geometric sum
            # I(eta) = sum_{m_lo}^{m_hi} e(m*eta) in closed form.
            if abs(eta) < 1e-15:
                I = complex(L, 0.0)
            else:
                z = np.exp(2j * np.pi * eta)
                I = z ** m_lo * (z ** L - 1) / (z - 1)
            main2 = (mu_int(Q) / phi_int(Q)) ** 2 * abs(I) ** 2
            errs.append(am2[j] - main2)
        errs = np.array(errs)
        rels.append(float(np.sqrt((errs ** 2).mean())) / float(L) ** 2)
    val = float(np.mean(rels))
    ok = val < 6e-3
    print(f"[sw face] T={T:.0f}: normalised RMS = {val:.3e} "
          f"{'PASS (< 6e-3)' if ok else 'FIRE'}")
    return ok


if __name__ == "__main__":
    face(float(sys.argv[1]) if len(sys.argv) > 1 else 9600.0)
