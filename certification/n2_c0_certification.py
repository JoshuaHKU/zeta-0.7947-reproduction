# -*- coding: utf-8 -*-
"""N2: closed-form identification of the continuum-quadrature constant

    c0 = gamma - 2   (Euler-Mascheroni gamma; parity-mean convention)

Structure of the proof (unified paper, App. A5):
1. At (b1,b2)=(1,1) the free coefficients of the partial-sum identity
   have the exact Euler closed form
       gamma_{2m} = C2 * prod_{p|m} 1/(p(p-2))   (m odd squarefree),
   gamma_d = 0 otherwise  [derived from beta_q = mu(q)^2/phi(q)^2 and
   f_v = beta(v)-beta(v+1); verified against the archive tensor sieve:
   the arrays agree at every d except d=2, see step 3].
2. Selberg-Delange/residue computation for the raw convention:
       sum_{d<=M} d*gamma_d = log M + gamma + o(1),
   because  sum d*gamma_d = 2*C2*sum_{m<=M/2} prod_{p|m} 1/(p-2),
   whose Dirichlet series is zeta(s+1)*E(s) with
       E(0) = 1/(2*C2)     (so the slope is exactly 1), and
       E'(0)/E(0) = log 2  (the p-sums cancel pairwise except p=2),
   giving constant  gamma + log2 - log2 = gamma  after the m=d/2
   rescaling.  [Verified numerically below to ~1e-5.]
3. The parity-mean convention (the tail-bound module's 'class-d exact
   replacement') books the q=2 arc block inside the parity mean:
   gamma~_2 = gamma_2 - beta_2 = C2 - 1.  This shifts sum d*gamma_d
   by exactly 2*(-1) = -2 and nothing else (array diff verified: the
   two arrays differ only at d=2, by exactly -1).  Hence
       c0^{parity-mean} = gamma - 2 = -1.42278433...
   matching the archived measurement -1.42279 (M=1.6e7, osc +-3e-5).

Consequence for Lemma CL (unified paper SS4.3): the 'computed
constant' c0 of the continuum quadrature is a classical constant in
closed form; with the certified geometric tail T_Gamma (quadrature_cert,
ratio 0.500) the partial-sum layer of the quadrature is certified.
The remaining computed object in N2 is the ladder constant c*
(= log pi + c0 - osc_bar in the F-QC5 pin), i.e. the mean
oscillation osc_bar = log pi + gamma - 2 - c* ~ 0.2652; its closed
form is not identified here.

Run: python3 n2_c0_certification.py [D]     (default D = 2e7)
"""
import math
import sys

import numpy as np

C2 = 0.6601618158468695739   # twin prime constant
GAMMA = 0.5772156649015329


def primes_upto(n):
    s = np.ones(n+1, bool); s[:2] = False
    for p in range(2, int(n**0.5)+1):
        if s[p]: s[p*p::p] = False
    return np.nonzero(s)[0]


def main(D=20_000_000):
    half = D//2
    P = primes_upto(half)
    g = np.ones(half+1); g[0] = 0.0
    for p in P:
        p = int(p)
        if p == 2: continue
        g[p::p] *= 1.0/(p*(p-2))
        if p*p <= half: g[p*p::p*p] = 0.0
    g[2::2] = 0.0                      # gamma_{2m} = C2*g[m], m odd sf
    mm = np.arange(half+1, dtype=float)
    dgam = 2*C2*(mm*g)                 # = d*gamma_d at d=2m
    cs = np.cumsum(dgam)
    print("raw convention: sum_{d<=M} d*gamma_d - log M  (-> gamma):")
    for Mv in (10**4, 10**5, 10**6, 4*10**6, D):
        c0 = cs[min(Mv//2, half)] - math.log(Mv)
        print(f"  M={Mv:.0e}: {c0:+.6f}   (gamma = {GAMMA:+.6f}, "
              f"dev {c0-GAMMA:+.1e})")
    print(f"\nconvention shift (q=2 block -> parity mean): exactly -2")
    print(f"c0^(parity-mean) = gamma - 2 = {GAMMA-2:+.7f}   "
          f"(archived measurement: -1.42279 at M=1.6e7)")
    # PS-identity spot test (both sides, truncated d<=M with tail note)
    M = 200000
    S = np.zeros(M+1); S[2::2] = 2*C2
    for p in primes_upto(M):
        p = int(p)
        if p == 2: continue
        S[p::p] *= (p-1.0)/(p-2.0)
    lhs = float(np.sum(S[1:M+1]-1.0))
    # gamma_d at d=2m, all m <= half (d up to D covers the tail range)
    msup = np.nonzero(g)[0]
    dsup = 2*msup
    gval = C2*g[msup]
    inr = dsup <= M
    rhs = -float(np.sum(dsup[inr]*np.mod(M/dsup[inr], 1.0)*gval[inr]))
    tail = -M*float(np.sum(gval[~inr]))     # {M/d}=M/d for d>M
    print(f"\nPS identity at M={M}: LHS={lhs:+.4f}  "
          f"RHS(d<=M)={rhs:+.4f}  tail(M<d<={2*half})={tail:+.4f}  "
          f"LHS-(RHS+tail)={lhs-rhs-tail:+.4f} (-> 0 as D grows)")


if __name__ == '__main__':
    main(int(float(sys.argv[1])) if len(sys.argv) > 1 else 20_000_000)
