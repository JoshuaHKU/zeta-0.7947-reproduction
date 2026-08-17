# -*- coding: utf-8 -*-
"""Shared utilities for the 13/18 reproduction pipeline.

Provides: prime tables, the von Mangoldt array, the twin singular
series S(h) = 2*C2*prod_{p|h, p odd}(p-1)/(p-2) (h even; 0 for h
odd), major-arc classification, and the two-sequence spectral
builder used by every numerical face in the paper.

All faces are pre-registered falsifiers: each module prints a PASS
or FIRE verdict against the criterion stated in its docstring.
"""
import math

import numpy as np

TWO_PI = 2.0 * math.pi
# Twin-prime constant C2 = prod_{p>2} (1 - 1/(p-1)^2).
C2_TWIN = 0.6601618158468696


def primes_upto(n):
    """Return all primes <= n (simple sieve of Eratosthenes)."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.nonzero(sieve)[0]


def lam_array(cap):
    """Von Mangoldt Lambda(n) for n = 0..cap as a float array.

    Lambda(p^k) = log p on prime powers, 0 elsewhere.
    """
    lam = np.zeros(cap + 1)
    for p in primes_upto(cap):
        q = p
        while q <= cap:
            lam[q] = math.log(p)
            q *= p
    return lam


def twin_singular_series(jmax):
    """S(h) for h = 0..jmax: the Hardy-Littlewood twin series.

    S(h) = 2*C2 * prod_{p | h, p odd} (p-1)/(p-2) for even h > 0,
    S(h) = 0 for odd h and for h = 0 (the h = 0 diagonal is
    handled exactly by the faces that need it).
    """
    s = np.zeros(jmax + 1)
    even = np.arange(2, jmax + 1, 2)
    s[even] = 2.0 * C2_TWIN
    for p in primes_upto(jmax):
        if p == 2:
            continue
        s[p::p] *= (p - 1.0) / (p - 2.0)
    s[1::2] = 0.0
    s[0] = 0.0
    return s


def arc_quality(gam, P, wid):
    """Major-arc quality of a frequency gam in [0,1).

    Returns the least Q <= P with |gam*Q - round(gam*Q)| <= wid
    (i.e. gam lies within wid/Q of a rational a/Q), or 0 if gam is
    minor at cutoff P.  The Q-independent right-hand side matches
    the archive convention (round 65, corrected classification).
    """
    for Q in range(1, P + 1):
        if abs(gam * Q - round(gam * Q)) <= wid:
            return Q
    return 0


def build_spectra(T, b1, b2):
    """Build the dilated two-sequence power spectra for a modulus
    pair (b1, b2), coprime.

    Windows: m in [X, 4X], n in [(b2/b1)X, (b2/b1)4X] with
    X = T/(2*pi); positions are the lattice points b2*m and b1*n,
    so the lock b1*n - b2*m = j is a position offset.  Returns
    (|S_m|^2, |S_n|^2, geometry dict); the FFT length N is padded
    to at least 4x the largest position so circular wraparound
    never touches the correlation ranges used by the faces.
    """
    X = int(T / TWO_PI)
    m_lo, m_hi = X, 4 * X
    n_lo, n_hi = (b2 * X) // b1, (4 * b2 * X) // b1
    lam = lam_array(max(m_hi, n_hi) + 10)
    pos_max = max(b2 * m_hi, b1 * n_hi)
    N = 1 << int(math.ceil(math.log2(4 * pos_max)))
    xm = np.zeros(N)
    xn = np.zeros(N)
    mm = np.arange(m_lo, m_hi + 1)
    nn = np.arange(n_lo, n_hi + 1)
    xm[b2 * mm] = lam[mm]
    xn[b1 * nn] = lam[nn]
    am2 = np.abs(np.fft.rfft(xm)) ** 2
    an2 = np.abs(np.fft.rfft(xn)) ** 2
    geo = dict(X=X, N=N,
               Ym=b2 * (m_hi - m_lo), Yn=b1 * (n_hi - n_lo),
               m_span=(m_lo, m_hi), n_span=(n_lo, n_hi))
    return am2, an2, geo


def mu_int(n):
    """Moebius function of a positive integer."""
    if n == 1:
        return 1
    r, p, m = 1, 2, n
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return 0
            r = -r
        p += 1
    return -r if m > 1 else r


def phi_int(n):
    """Euler totient of a positive integer."""
    r, p, m = 1, 2, n
    while p * p <= m:
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            r *= (p - 1) * p ** (e - 1)
        p += 1
    if m > 1:
        r *= m - 1
    return r
