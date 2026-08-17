# -*- coding: utf-8 -*-
"""Face 4 -- the discrete Parseval bridge (display (6.2)).

The lock-variance equals the spectral L2 distance:
sum_tau rho(tau)^2 = (1/N) * [dens_0^2 + 2*sum dens^2 + dens_Nyq^2].
This is an exact identity; the face verifies it bitwise on the
actual spectra.  PASS: relative difference below 1e-10.
"""
import sys

import numpy as np

from common import build_spectra


def face(T):
    ok = True
    for (b1, b2) in [(5, 3), (3, 2)]:
        am2, an2, geo = build_spectra(T, b1, b2)
        N = geo['N']
        dens = am2 * an2
        full = np.zeros(N // 2 + 1)
        full[:len(dens)] = dens
        rho = np.fft.irfft(full, n=N)
        lhs = float((rho ** 2).sum())
        rhs = (float(full[0] ** 2)
               + 2.0 * float((full[1:-1] ** 2).sum())
               + float(full[-1] ** 2)) / N
        rel = abs(lhs - rhs) / lhs
        good = rel < 1e-10
        ok &= good
        print(f"  ({b1},{b2}): relative difference {rel:.2e} "
              f"{'PASS' if good else 'FIRE'}")
    print(f"[parseval face] {'ALL PASS' if ok else 'FIRED'}")
    return ok


if __name__ == "__main__":
    face(float(sys.argv[1]) if len(sys.argv) > 1 else 9600.0)
