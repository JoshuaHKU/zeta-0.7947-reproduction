# -*- coding: utf-8 -*-
"""Face LP-full -- the three-moment consumption with correlated
bands (the theorem's linear program).

C5 = 0.0278 + delta, |delta| <= 0.0001 enters m5 (x1) and m6
(x6); independent m6 band 0.0018 (midpoint-ladder corrected
constants, paper §5.5 and D10).  Scans the worst case over
delta.  PASS: certified edge reproduces 0.79472/0.89736 within
5e-4, grid-stable.
"""
import numpy as np
from scipy.optimize import linprog


def solve(m5, m6, dx=0.002):
    xg = np.arange(0, 10 + 1e-9, dx)
    Aeq = np.array([np.ones_like(xg)] + [xg ** k for k in range(1, 5)])
    r = linprog(-(xg < 1e-12).astype(float),
                A_ub=np.array([-xg ** 5, xg ** 6]), b_ub=[-m5, m6],
                A_eq=Aeq, b_eq=[1.0, 1, 4 / 3, 2, 3.25],
                bounds=[(0, None)] * len(xg), method='highs')
    return (1 + 2 * r.fun, 1 + r.fun) if r.success else (0.0, 0.0)


def face():
    ok = True
    for dx in (0.002, 0.001):
        worst = (1.0, 1.0)
        for d in np.linspace(-0.0001, 0.0001, 5):
            m5 = 67 / 12 + 0.0278 + d
            m6 = (9.7500 + 6 * (0.0278 + d) + 0.3119048 - 0.0552
                  - 0.0078 + 0.0018)
            s = solve(m5, m6, dx)
            if s[0] < worst[0]:
                worst = s
        good = abs(worst[0] - 0.79472) < 5e-4
        ok &= good
        print(f"  dx={dx}: certified edge {worst[0]:.5f} / "
              f"{worst[1]:.5f} {'PASS' if good else 'FIRE'}")
    print(f"[lp-full face] {'ALL PASS' if ok else 'FIRED'}")
    return ok


if __name__ == "__main__":
    face()
