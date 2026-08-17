# -*- coding: utf-8 -*-
"""Re-optimized certificate threshold (paper §3 remark): maximizing
the origin mass over measures with (m1,m2,m3) pinned and m4 <= Mbar,
per Mbar; reproduces the record-breaking budget 0.0721.
证书再优化阈值 (论文 §3 注记): 在 (m1,m2,m3) 钉定、m4 ≤ Mbar 约束下
对原点质量做 LP 极大化; 复现破纪录预算 0.0721.
"""
import numpy as np
from scipy.optimize import linprog
xg = np.arange(0.0, 5.0001, 0.002); n = len(xg)
A_eq = np.vstack([xg**0, xg**1, xg**2, xg**3]); b_eq = np.array([1.0,1.0,4/3,2.0])
obj = np.zeros(n); obj[0] = -1.0   # mass exactly at 0 (theta -> 0+)
def w0_of(Mb):
    r = linprog(obj, A_ub=(xg**4)[None,:], b_ub=[Mb+1e-9], A_eq=A_eq, b_eq=b_eq,
                bounds=[(0,None)]*n, method='highs')
    return None if r.x is None else -r.fun
print("Mbar      w0        constant")
for Mb in [13/4, 13/4*1.01, 13/4*1.02, 13/4*1.03, 13/4*1.05, 13/4*1.10, 13/4*1.20]:
    w0 = w0_of(Mb)
    print(f"{Mb:.4f}  {w0:.5f}   {1-2*w0:.5f}")
print(f"exact calibration: 5/36 = {5/36:.5f} -> 13/18 = {13/18:.5f}")
lo, hi = 13/4, 13/4*2
for _ in range(45):
    mid = (lo+hi)/2
    w0m = w0_of(mid)
    if w0m is not None and 1-2*w0m > 0.6725: lo = mid
    else: hi = mid
print(f"M* = {lo:.4f} = (13/4) x {lo/(13/4):.4f}   -> band excess budget = {lo-13/4:.4f} abs = {100*(lo/(13/4)-1):.2f}%")
