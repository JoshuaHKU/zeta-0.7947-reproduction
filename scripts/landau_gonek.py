import numpy as np, math
from scipy.optimize import linprog

# corrected moments: (m1,m2,m3,m4) = (1, 4/3, 2, 13/4); Hankel-valid (mu4 = 1/4 > mu2^2=1/9 ok)
xg = np.arange(0.0, 6.0001, 0.004); n = len(xg)
A_eq = np.vstack([xg**0, xg**1, xg**2, xg**3])
b_eq = np.array([1.0, 1.0, 4/3, 2.0])
obj = -(xg <= 0.02).astype(float)
def w0_of(Mbar):
    r = linprog(obj, A_ub=(xg**4)[None,:], b_ub=[Mbar], A_eq=A_eq, b_eq=b_eq,
                bounds=[(0,None)]*n, method='highs')
    return None if r.x is None else -r.fun
print("Mbar       w0        constant=1-2w0")
for Mb in [13/4, 13/4*1.02, 13/4*1.05, 13/4*1.10, 13/4*1.20, 13/4*1.5, 13/4*2.5]:
    w0 = w0_of(Mb)
    print(f"{Mb:.4f}   {'INFEAS' if w0 is None else f'{w0:.5f}   {1-2*w0:.5f}'}")
print(f"calibration target: w0=5/36={5/36:.5f}, constant=13/18={13/18:.5f}")
lo, hi = 13/4, 13/4*4
if w0_of(hi) is not None and 1-2*w0_of(hi) < 0.6725:
    for _ in range(40):
        mid = (lo+hi)/2
        if 1-2*w0_of(mid) > 0.6725: lo = mid
        else: hi = mid
    print(f"M* (beat 0.6725): m4-upper < {lo:.4f} = 13/4 x {lo/(13/4):.4f}  "
          f"(band excess budget {100*(lo/(13/4)-1):.1f}%)")

# Landau-Gonek first strike
gam = np.load('ck_zeros.npy'); T0 = 1950.0
g = gam[gam <= T0]; N0 = len(g)
print(f"\nLandau-Gonek engine calibration: {N0} zeros, T0={T0:.0f}")
print(" X    data Sum|A|^2     diag pred     offdiag(Landau)    total       data/pred")
for X in [50, 100, 200]:
    lv = np.zeros(X+1)
    for p in range(2, X+1):
        if all(p % q for q in range(2, int(p**0.5)+1)):
            pk = p
            while pk <= X: lv[pk] = math.log(p); pk *= p
    ns = np.array([m for m in range(2, X+1) if lv[m] > 0])
    w = lv[ns]/np.sqrt(ns)
    Av = np.exp(-1j*np.outer(g, np.log(ns))) @ w
    data = float(np.sum(np.abs(Av)**2))
    diag = N0*float(np.sum(lv[ns]**2/ns))
    offd = 0.0
    for p in [q for q in range(2, X+1) if all(q % r for r in range(2, int(q**0.5)+1))]:
        lp = math.log(p); c = 2
        while p**c <= X:
            offd -= (T0/math.pi)*lp**3*(c-1)*p**(-c); c += 1
    print(f"{X:4d}   {data:12.1f}  {diag:12.1f}  {offd:+11.1f}  {diag+offd:11.1f}   {data/(diag+offd):.4f}")
