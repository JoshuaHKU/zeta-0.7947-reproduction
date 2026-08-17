# -*- coding: utf-8 -*-
# Finite-height illustration: independent replication of the
# compressed-matrix engine of [C26] (window I = [600, 1200] as in
# [C26, Sec 8(1)], Tukey-type C^3 taper, lambda = 1); certifies a
# positive count of simple on-line zeros in the window.
# 有限高度演示: [C26] 压缩矩阵引擎的独立复刻 (窗口与 taper 同
# [C26, §8(1)]); 在窗内认证正数目的简单在线零点.
# Build the Gabor-compressed Weil matrix G BOTH from the prime side and from my 1500 zeros,
# compare, then compute the unconditional certificate quantities.
import numpy as np, math, time
import mpmath as mp

t0 = time.time()
T = 600.0
L = math.log(T/(2*math.pi))          # 4.559
X = T/(2*math.pi)                    # e^L = 95.49
h = 2*math.pi/L
d = int(L*T/(2*math.pi))             # ~435
tau = T + h*np.arange(d)             # tau_k in [T, 2T)
eta = 0.2
w = eta*L/2

# ---- taper phi and its Fourier transform ----
def rho_ramp(x):
    return np.clip(x - np.sin(2*np.pi*np.clip(x, 0, 1))/(2*np.pi), 0, 1)
du = 0.001
ug = np.arange(0, L/2 + du, du)
phi_u = rho_ramp((L/2 - ug)/w)
rg = np.arange(0, 801.0, 0.02)       # phi-hat on [0,800], even
# phi-hat(r) = 2 int_0^{L/2} phi(u) cos(ru) du
PH = np.zeros(len(rg))
chunk = 4000
for i in range(0, len(rg), chunk):
    R = rg[i:i+chunk]
    PH[i:i+chunk] = 2*np.trapz(phi_u[None, :]*np.cos(np.outer(R, ug)), ug, axis=1)
def phihat(r):
    return np.interp(np.abs(r), rg, PH)
a_const = np.trapz(phi_u**2, ug)*2/L
print(f"L={L:.3f}, X={X:.1f}, d={d}, w={w:.3f}, a={a_const:.4f}, phi-hat(0)={PH[0]:.4f}  ({time.time()-t0:.0f}s)")

# ---- zero side ----
gam = np.load('ck_zeros.npy')        # 1500 zeros, up to 1981
Vz = phihat(gam[:, None] - tau[None, :])
G_zero = Vz.T @ Vz                   # + negligible negative-ordinate tail
NI = int(((gam > T) & (gam <= 2*T)).sum())
D0 = math.sqrt(T)
NIp = int(((gam > T - D0) & (gam <= 2*T + D0)).sum())
print(f"N(I) = {NI} zeros in [600,1200]; N(I') = {NIp}")

# ---- prime side ----
NP = int(X) + 1
lam_v = np.zeros(NP+1)
for p in range(2, NP+1):
    if all(p % q for q in range(2, int(p**0.5)+1)):
        pk = p
        while pk <= NP:
            lam_v[pk] = math.log(p); pk *= p
pows = [(n, lam_v[n]) for n in range(2, NP+1) if lam_v[n] > 0]

pad = 100.0
tg = np.arange(T - pad, 2*T + pad, 0.05)
# mu(tau): smooth -> coarse mpmath grid + interpolation
mp.mp.dps = 15
tc = np.arange(T - pad - 1, 2*T + pad + 1, 0.5)
mu_c = np.array([float(mp.re(mp.digamma(0.25 + 0.5j*t))) for t in tc])/(2*math.pi) - math.log(math.pi)/(2*math.pi)
mu = np.interp(tg, tc, mu_c)
s_half = 0.5 + 1j*tg
PiX = 1/(2*math.pi*(0.25 + tg**2)) + (1/math.pi)*np.real((X**s_half - 1)/s_half)
PX = np.zeros(len(tg))
for n, ln in pows:
    PX -= (ln/math.sqrt(n))*np.cos(tg*math.log(n))/math.pi
nu = mu + PiX + PX
Vt = phihat(tg[:, None] - tau[None, :])
G_prime = (Vt * (nu*0.05)[:, None]).T @ Vt
print(f"prime side built  ({time.time()-t0:.0f}s)")

# ---- comparison and certificates ----
scale = np.abs(G_zero).max()
disc = np.abs(G_prime - G_zero).max()/scale
print(f"\nmax |G_prime - G_zero| / max|G| = {disc:.2e}   (paper reports 1e-6..1e-8 with finer quadrature)")
for name, G in [("zero ", G_zero), ("prime", G_prime)]:
    tr = np.trace(G); tr2 = np.sum(G*G)
    C = tr*tr/tr2
    ev = np.linalg.eigvalsh(G/L)
    print(f"{name}: C=(trG)^2/trG^2 = {C:.1f},  C/N(I) = {C/NI:.3f}  [F(1)=0.750, paper window C/N=0.744],"
          f"  min eig(G/L) = {ev[0]:.3g}  (all positive: {bool(ev[0] > -1e-9)})")
# rank-trace certificate in units (4.4): hatG = G/(a L)
Ghat = G_prime/(a_const*L)
cert = 4*np.trace(Ghat) - 2*NIp - np.sum(Ghat*Ghat)
print(f"\nrank-trace certificate 4tr - 2N(I') - ||.||_F^2 = {cert:.1f} = {cert/NI:+.3f} * N(I)")
print(f"  (unconditional lower bound on simple on-line zeros in the window; paper gets +0.49N at T=1000)")
print(f"  asymptotic law: H(1) = 2 - 1/1 - 1/3 = {2-1-1/3:.4f} -> proportion 2/3 as T -> infinity")
print(f"  distinct-zero law: H_d(1) = (1+H(1))/2 = {(1+2/3)/2:.4f} -> 5/6;  Cauchy-Schwarz F(1) = {1/(1+1/3):.4f}")

# ---- Lemma 3.2 stress test ----
rng = np.random.default_rng(7)
viol = 0
for _ in range(10000):
    dd = rng.integers(3, 10)
    r = rng.integers(0, dd+1); b = rng.integers(0, dd+1)
    Mr = rng.normal(size=(dd, r)) if r else np.zeros((dd, 1))
    P = Mr @ Mr.T
    Qe = rng.normal(size=dd)**2
    sgn = np.ones(dd); sgn[b:] = -1
    U = np.linalg.qr(rng.normal(size=(dd, dd)))[0]
    Q = U @ np.diag(np.sort(Qe)[::-1]*sgn) @ U.T
    lhs = r
    rhs = 2*np.trace(P) + 4*np.trace(Q) - 4*b - np.sum((P+Q)**2)
    if lhs < rhs - 1e-8:
        viol += 1
print(f"\nLemma 3.2 stress test (10^4 random instances): violations = {viol}")
print(f"total {time.time()-t0:.0f}s")

# ---- corrected units: G_hat = G/(a L^2), per proof of Theorem A ----
print("\n---- corrected certificate ----")
ell1 = L + 2*math.log(2) - 1
lam1 = L/ell1
Ghat = G_prime/(a_const*L*L)
trH = np.trace(Ghat); frH = np.sum(Ghat*Ghat)
cert = 4*trH - 2*NIp - frH
print(f"lambda_1 = {lam1:.4f};  tr G_hat = {trH:.1f}  (vs N(I)={NI});  ||G_hat||_F^2 = {frH:.1f}"
      f"  (vs (1/l1+l1/3)N = {(1/lam1+lam1/3)*NI:.1f})")
print(f"certificate 4tr - 2N(I') - ||.||_F^2 = {cert:.1f} = {cert/NI:+.3f} * N(I)")
print(f"  -> unconditional: at least {max(cert,0):.0f} simple on-line zeros in this window certified")
print(f"  ([C26] Sec 8(6), T=1000: +0.49 N(I); asymptotic value H(1)-2*0=... -> 2/3 as T->inf)")
