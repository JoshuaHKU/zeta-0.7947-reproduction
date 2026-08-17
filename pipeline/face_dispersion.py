# -*- coding: utf-8 -*-
"""Face 3 -- the tau-resolved model-tracking and dispersion face.

Builds the lock-resolved four-point profile rho(tau) =
irfft(|S_m|^2 |S_n|^2) and its singular-series model
Model(tau) = sum over the CRT progression t = 0 mod b2,
t = tau mod b1 of S(t/b2) S((t-tau)/b1) Vm Vn plus the exact
diagonal pieces.  Reports (i) the tracking ratio
RMS(resid)/RMS(model) and (ii) the normalised dispersion
RMS(resid)/rho(0), averaged over sixteen coprime modulus pairs.
Archive trend: dispersion follows the pure averaging law T^{-1/2}
over a 64x range with tracking to 0.2% at the top.  PASS: median
tracking below 4% at the requested height.
"""
import sys

import numpy as np

from common import build_spectra, twin_singular_series

FAMS = [(5, 3), (5, 7), (3, 2), (7, 3), (7, 2), (7, 5),
        (4, 3), (9, 2), (8, 3), (9, 4), (5, 2), (8, 5),
        (9, 5), (8, 7), (9, 7), (7, 4)]


def profile_and_model(T, b1, b2):
    """Return rho(tau), Model(tau), rho(0), lock range J."""
    am2, an2, geo = build_spectra(T, b1, b2)
    N, X = geo['N'], geo['X']
    dens = am2 * an2
    full = np.zeros(N // 2 + 1)
    full[:len(dens)] = dens
    rho = np.fft.irfft(full, n=N)
    J = max(8, X // 2)
    m_lo, m_hi = geo['m_span']
    n_lo, n_hi = geo['n_span']
    from common import lam_array
    lam = lam_array(max(m_hi, n_hi) + 10)
    Am0 = float((lam[np.arange(m_lo, m_hi + 1)] ** 2).sum())
    An0 = float((lam[np.arange(n_lo, n_hi + 1)] ** 2).sum())
    Km, Kn = m_hi - m_lo, n_hi - n_lo
    s = twin_singular_series(max(Km, Kn) + 10)

    def Vm(u):
        return np.maximum(Km + 1 - np.abs(u), 0)

    def Vn(u):
        return np.maximum(Kn + 1 - np.abs(u), 0)

    inv = pow(b2, -1, b1) if b1 > 1 else 0
    M = b1 * b2
    tmax = b2 * Km
    model = np.zeros(J)
    for tau in range(1, J):
        # CRT: t = 0 (mod b2) and t = tau (mod b1).
        t0 = (b2 * ((tau * inv) % b1)) % M
        ts = np.arange(t0 - ((t0 + tmax) // M) * M, tmax + 1, M)
        ts = ts[np.abs(ts) <= tmax]
        u = ts // b2
        w = (ts - tau) // b1
        ok = (ts != 0) & (ts != tau) & (np.abs(w) <= Kn) & (u != 0)
        u_, w_ = u[ok], w[ok]
        val = float((s[np.abs(u_)] * s[np.abs(w_)]
                     * Vm(u_) * Vn(w_)).sum())
        # Exact diagonal pieces t = 0 and t = tau.
        if tau % b1 == 0:
            val += Am0 * s[abs(tau) // b1] * float(Vn(tau // b1))
        if tau % b2 == 0:
            val += An0 * s[abs(tau) // b2] * float(Vm(tau // b2))
        model[tau] = val
    return rho, model, float(rho[0]), J


def face(T):
    tracks, disps = [], []
    for (b1, b2) in FAMS:
        rho, model, rho0, J = profile_and_model(T, b1, b2)
        resid = rho[1:J] - model[1:J]
        rms_r = float(np.sqrt((resid ** 2).mean()))
        rms_m = float(np.sqrt((model[1:J] ** 2).mean()))
        tracks.append(rms_r / rms_m)
        disps.append(rms_r / rho0)
    tr, dp = float(np.median(tracks)), float(np.mean(disps))
    ok = tr < 0.04
    print(f"[dispersion face] T={T:.0f}: tracking median = {tr:.4f}"
          f" {'PASS (< 4%)' if ok else 'FIRE'}; "
          f"normalised dispersion = {dp:.3e}")
    return ok


if __name__ == "__main__":
    face(float(sys.argv[1]) if len(sys.argv) > 1 else 9600.0)
