# -*- coding: utf-8 -*-
# Numerical gates for the third-moment ledger (paper §3):
# (a) Sum_{n<=X} (Lambda(n)^2/n)(L - log n) = L^3/6 (1+O(1/L))
# (b) int_T^{2T} log^3(tau/2pi) dtau vs T*l^3 expansion (mu^3 main term)
# (c) proportion dividends from the measured/model R class
# 第三矩账本的数值门 (论文 §3): (a) L³/6 恒等式; (b) μ³ 类主项;
# (c) R 类实测/模型的比例红利.
import numpy as np, math

def lam_sieve(N):
    lam = np.zeros(N+1)
    for p in range(2, N+1):
        if all(p % q for q in range(2, int(p**0.5)+1)):
            pk = p
            while pk <= N:
                lam[pk] = math.log(p); pk *= p
    return lam

print("(a) S(X) = Sum (Lambda^2/n)(L-log n) vs L^3/6:")
for X in [10**4, 10**5, 10**6]:
    L = math.log(X); lam = lam_sieve(X)
    n = np.arange(1, X+1); m = lam[1:] > 0
    S = np.sum(lam[1:][m]**2/n[m]*(L - np.log(n[m])))
    print(f"  X=10^{int(math.log10(X))}: S={S:.3f}, L^3/6={L**3/6:.3f}, ratio={S/(L**3/6):.4f}")

print("\n(b) mu^3 log-moment: int_T^2T log^3(t/2pi) dt / (T l^3), l=log(T/2pi):")
for T in [600., 38400., 1e8]:
    tg = np.linspace(T, 2*T, 400001)
    I = np.trapz(np.log(tg/(2*math.pi))**3, tg)
    l = math.log(T/(2*math.pi))
    # ell1 = l + 2log2 - 1 is the exact first moment; cube comparison:
    ell1 = l + 2*math.log(2) - 1
    print(f"  T={T:.0f}: I/(T l^3)={I/(T*l**3):.4f}, I/(T ell1^3)={I/(T*ell1**3):.4f}  (→1 slowly; 1+O(1/l) as claimed)")

print("\n(c) proportion dividends (kappa = 13/18 - 8c/9, c = 0.01767 + c_H):")
for tag, R in [("c_H=0 boundary", 0.0), ("measured l=8.72", -0.01006), ("model", -0.0166)]:
    c = 0.01767 + R
    print(f"  {tag}: c={c:+.5f}, kappa_simple={13/18-8*c/9:.5f}, kappa_distinct={31/36-4*c/9:.5f}")
print(f"  record-breaking threshold: c_H < {0.72222-0.6725 - 0.0}*9/8 - 0.01767 = {(13/18-0.6725)*9/8-0.01767:.5f}")
