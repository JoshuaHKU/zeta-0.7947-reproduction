import numpy as np
from scipy.optimize import linprog
xg = np.arange(0.0, 8.0001, 0.002); n = len(xg)
obj = np.zeros(n); obj[0] = -1.0
def w0(moms_eq, top_ub=None):
    A_eq = np.vstack([xg**k for k in range(len(moms_eq))])
    b_eq = np.array(moms_eq)
    A_ub = None; b_ub = None
    if top_ub is not None:
        k = len(moms_eq)
        A_ub = (xg**k)[None,:]; b_ub = [top_ub]
    r = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0,None)]*n, method='highs')
    return None if r.x is None else -r.fun

m5, m6 = 5.5733, 10.0207; s5, s6 = 0.1132, 0.2692
print("ladder (lambda->1): constant_simple = 1-2w0, constant_distinct = 1-w0")
cases = [
 ("k<=2  (baseline)",            [1,1,4/3], None),
 ("k<=4  (calibration 13/18)",   [1,1,4/3,2], 13/4),
 ("k<=5  (m5 equality)",         [1,1,4/3,2,13/4], m5),
 ("k<=6  (central values)",      [1,1,4/3,2,13/4,m5], m6),
 ("k<=6  (m5+s, m6+s: worst)",   [1,1,4/3,2,13/4,m5+s5], m6+s6),
 ("k<=6  (m5-s, m6-s: best)",    [1,1,4/3,2,13/4,m5-s5], m6-s6),
]
for name, eqs, ub in cases:
    w = w0(eqs, ub)
    if w is None: print(f"{name}:  INFEASIBLE"); continue
    print(f"{name}:  w0={w:.5f}  simple={1-2*w:.5f}  distinct={1-w:.5f}")
# budget: how much m6-excess before falling back to 13/18-level?
lo, hi = m6, m6*2
for _ in range(40):
    mid = (lo+hi)/2
    w = w0([1,1,4/3,2,13/4,m5], mid)
    if w is not None and 1-2*w > 13/18: lo = mid
    else: hi = mid
print(f"\nM6* (stay above 13/18): m6-upper < {lo:.3f} = m6 + {lo-m6:.3f}  "
      f"({100*(lo/m6-1):.1f}% excess budget)")
