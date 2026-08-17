# -*- coding: utf-8 -*-
"""{4,2} 旁观对阶梯的 Romberg 外推 + 有理重构。
Romberg extrapolation + rational reconstruction for the {4,2}
spectator-pair ladder (paper §5.5; gate G4 of
spectator_42_reference.py).

对 U₁, U₂, U₃ 和总值 6U₁+6U₂+3U₃ 各做一次 h² 阶 Romberg，
列选择与误差带的取法同 analyze_ladder.py：对每一列用最后两行之差
估计收敛误差，取误差最小的列；重构分母上限取
qmax = ⌊√(1/(2·eps))⌋，使「唯一候选」成为定理而非观察。

Usage: python3 analyze_42.py results/res_42.txt
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from analyze_ladder import geo_tail, romberg, best_column
from rational_reconstruct import candidates

PREREG = Fraction(-23, 420)          # F-RAT-42 预注册候选


def read42(path):
    """dv -> (U1, U2, U3, total, wall, device)."""
    rows = {}
    for line in open(path):
        p = line.split()
        if len(p) < 10 or p[0] != "42":
            continue
        rows[float(p[1])] = (float(p[4]), float(p[5]), float(p[6]),
                             float(p[7]), float(p[8]), " ".join(p[9:]))
    return sorted(rows.items(), key=lambda kv: -kv[0])


def ladder(rows, k):
    return [(dv, r[k]) for dv, r in rows]


def do(name, vals, prereg=None):
    vals = geo_tail(vals)
    T = romberg(vals)
    val, err, col = best_column(T)
    band = 10.0 * err
    qmax = max(2, int((1.0 / (2.0 * band)) ** 0.5))
    print(f"\n--- {name} ---")
    print("  dv           raw                    Richardson j=1..3")
    for k, (dv, v) in enumerate(vals):
        row = "  ".join(f"{x:+.13f}" for x in T[k][1:4])
        print(f"  {dv:<11g} {v:+.15f}   {row}")
    print(f"  best column j={col}  value {val:+.15f}  drift {err:.2e}  "
          f"band {band:.1e}  qmax {qmax}")
    cs = candidates(val, band, qmax)
    for f, d in cs[:5]:
        star = "  <== 预注册 pre-registered" if prereg is not None and f == prereg else ""
        print(f"  candidate {str(f):>16s} = {float(f):+.15f}  |dev| = {d:.2e}{star}")
    if len(cs) == 1:
        print(f"  ==> UNIQUE: {cs[0][0]}")
    elif not cs:
        print("  ==> no candidate in band at the guaranteed qmax")
    return val, band, cs


def ratio_check(vals, target):
    """不经 Romberg 的独立检验：原始阶梯值相对分数的误差比 → 4.000?"""
    print(f"\n--- 独立检验 / independent check vs {target} ---")
    print("  dv           e(dv) = ladder - p/q      e(2h)/e(h)")
    prev = None
    for dv, v in geo_tail(vals):
        e = v - float(target)
        r = f"{prev/e:8.4f}" if prev is not None and e != 0 else "    —   "
        print(f"  {dv:<11g} {e:+.6e}          {r}")
        prev = e


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/res_42.txt"
    rows = read42(path)
    print("=" * 72)
    print("{4,2} 旁观对变体 — 各档原值 / per-rung values")
    print("=" * 72)
    print("| dv | U1 | U2 | U3 | {4,2}=6U1+6U2+3U3 | wall | device |")
    print("|---|---|---|---|---|---|---|")
    for dv, r in rows:
        print(f"| {dv:g} | {r[0]:+.12f} | {r[1]:+.12f} | {r[2]:+.12f} | "
              f"{r[3]:+.12f} | {r[4]:.1f} s | {r[5]} |")
    for k, nm in ((0, "U1"), (1, "U2"), (2, "U3")):
        do(nm, ladder(rows, k))
    val, band, cs = do("{4,2} = 6U1+6U2+3U3", ladder(rows, 3), PREREG)
    ratio_check(ladder(rows, 3), PREREG)
    print(f"\n预注册候选 F-RAT-42 = {PREREG} = {float(PREREG):+.15f}")
    print(f"本次外推           = {val:+.15f}   dev = {abs(val-float(PREREG)):.2e}")
    print(f"论文当前带          = -0.0552(8)  -> 包含? "
          f"{'YES' if abs(val + 0.0552) <= 0.0008 else 'NO'}")
