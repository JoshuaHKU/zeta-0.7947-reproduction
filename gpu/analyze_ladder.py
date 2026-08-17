# -*- coding: utf-8 -*-
"""Romberg 外推 + 有理重构 / Romberg extrapolation + rational
reconstruction for the midpoint-ladder runs (paper §5.5, item N1).

阶梯误差在 b=4 上实测为干净的偶次幂展开 (相邻误差比 3.964 → 3.991
→ 3.998 → 4.000 → 4.000), 故 h=dv 的 h² 阶 Romberg 表适用:
    T[k][j] = (4^j·T[k][j-1] - T[k-1][j-1]) / (4^j - 1)

列选择 / column selection: 最粗的几档尚未进入渐近区, 会污染高阶列,
因此不取整条对角线, 而是对每一列 j 用最后两行之差估计其收敛误差,
取误差最小的列作为最终值.  The coarsest rungs are not yet asymptotic
and pollute the high-order columns, so instead of the full diagonal we
score each column j by |T[N-1][j] - T[N-2][j]| and take the best.

重构界 / reconstruction bound: 两个分母 ≤ Q 的不同分数若都落在 x 的
eps 邻域内, 则 Q² ≥ 1/(2·eps); 故取 Q = floor(sqrt(1/(2·eps))) 时
候选必唯一 (若存在).  Uniqueness is then guaranteed, not assumed.

Usage:  python3 analyze_ladder.py res_b5.txt [res_b6.txt ...]
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from rational_reconstruct import candidates


def read(path):
    """(b, [(dv, value)]) rungs, deduplicated by dv, coarse -> fine."""
    rows, b = {}, None
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        b = int(p[0])
        rows[float(p[1])] = float(p[4])
    return b, sorted(rows.items(), key=lambda kv: -kv[0])


def geo_tail(vals):
    """Longest ratio-2 geometric tail ending at the finest rung."""
    geo = [vals[-1]]
    for dv, v in reversed(vals[:-1]):
        if abs(dv / geo[0][0] - 2.0) < 1e-9:
            geo.insert(0, (dv, v))
        else:
            break
    return geo


def romberg(vals):
    T = [[v] for _, v in vals]
    for j in range(1, len(T)):
        for k in range(j, len(T)):
            f = 4.0 ** j
            T[k].append((f * T[k][j - 1] - T[k - 1][j - 1]) / (f - 1.0))
    return T


def best_column(T):
    """(value, band, column) — column with the smallest row-to-row drift."""
    N = len(T)
    best = None
    for j in range(0, N - 1):
        if j >= len(T[N - 2]):
            break
        err = abs(T[N - 1][j] - T[N - 2][j])
        err = max(err, 2e-16)
        if best is None or err < best[1]:
            best = (T[N - 1][j], err, j)
    if best is None:
        return T[-1][-1], 1.0, 0
    return best


def analyze(path, exact=None, verbose=True):
    b, allvals = read(path)
    vals = geo_tail(allvals)
    T = romberg(vals)
    val, err, col = best_column(T)
    band = 10.0 * err                      # conservative 10x safety factor
    qmax = max(2, int((1.0 / (2.0 * band)) ** 0.5))
    if verbose:
        print(f"\n=== b = {b}   ({len(vals)} rungs, dv "
              f"{vals[0][0]:g} .. {vals[-1][0]:g}) ===")
        print("  dv           raw ladder value      Richardson columns "
              "T[last][j] ->")
        for k, (dv, v) in enumerate(vals):
            row = "  ".join(f"{x:+.12f}" for x in T[k][1:4])
            print(f"  {dv:<11g} {v:+.15f}   {row}")
        print(f"  best column j={col}  value {val:+.15f}  "
              f"drift {err:.2e}  band {band:.1e}  qmax {qmax}")
    if exact is not None:
        print(f"  exact = {float(exact):+.15f}   "
              f"dev = {abs(val-float(exact)):.2e}   [CALIBRATION]")
    cs = candidates(val, band, qmax)
    for f, d in cs[:6]:
        print(f"  candidate {str(f):>18s} = {float(f):+.15f}  |dev| = {d:.2e}")
    if len(cs) == 1:
        print(f"  ==> UNIQUE rational reconstruction: {cs[0][0]}")
    elif not cs:
        print("  ==> no rational candidate in band at the guaranteed qmax")
    return b, val, band, qmax, cs, vals, T


if __name__ == "__main__":
    for path in sys.argv[1:]:
        ex = None
        if path.endswith("b4.txt"):
            ex = Fraction(-1, 60)
        if path.endswith("b3.txt"):
            ex = Fraction(0)
        try:
            analyze(path, ex)
        except Exception as e:
            print(f"{path}: {type(e).__name__}: {e}")
