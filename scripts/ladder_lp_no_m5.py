import numpy as np
from scipy.optimize import linprog
xg = np.arange(0.0, 8.0001, 0.002); n = len(xg)
obj = np.zeros(n); obj[0] = -1.0
def w0(eq_ks, eq_vals, ub_k=None, ub_val=None):
    A_eq = np.vstack([xg**k for k in eq_ks]); b_eq = np.array(eq_vals)
    A_ub = (xg**ub_k)[None,:] if ub_k else None
    b_ub = [ub_val] if ub_k else None
    r = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0,None)]*n, method='highs')
    return None if r.x is None else -r.fun
m5c, m6c = 5.5733, 10.0207
# variant: drop m5 entirely (no odd-moment evaluation needed in the NT cascade)
w = w0([0,1,2,3,4],[1,1,4/3,2,13/4], 6, m6c)
print(f"k<=6 WITHOUT m5:  w0={w:.5f}  simple={1-2*w:.5f}  distinct={1-w:.5f}")
w2 = w0([0,1,2,3,4,5],[1,1,4/3,2,13/4,m5c], 6, m6c)
print(f"k<=6 with m5:     w0={w2:.5f}  simple={1-2*w2:.5f}  distinct={1-w2:.5f}")
# sensitivity of the no-m5 constant to the m6 upper bound
for M6 in [m6c, 12, 15, 20, 25]:
    ww = w0([0,1,2,3,4],[1,1,4/3,2,13/4], 6, M6)
    print(f"  m6-upper = {M6:6.2f}:  simple = {1-2*ww:.5f}")
# m6 feasibility floor given m1..m4 (context)
lo, hi = 3.0, m6c
for _ in range(40):
    mid=(lo+hi)/2
    ww = w0([0,1,2,3,4],[1,1,4/3,2,13/4], 6, mid)
    if ww is None: lo=mid
    else: hi=mid
print(f"m6 feasibility floor given m1..m4: {hi:.3f}  (true m6 = {m6c:.2f})")
