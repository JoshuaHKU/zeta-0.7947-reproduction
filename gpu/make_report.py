# -*- coding: utf-8 -*-
"""由原始阶梯数据装配报告的结果表 (Markdown)。
Assemble the report's results tables from the raw rung files.

Usage:  python3 make_report.py [results_dir]   > section.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from analyze_ladder import read, geo_tail, romberg, best_column
from rational_reconstruct import candidates

NAME = {3: "∫O₃C₃", 4: "Φ₄ = ∫O₄C₄", 5: "C₅ = ∫O₅C₅",
        6: "{6} = ∫O₆C₆", 7: "C₇ = ∫O₇C₇"}
PAPER = {3: "0 (已知 known)", 4: "−1/60 (精确 exact)", 5: "0.0278(1)",
         6: "−0.0078(8)", 7: "— (侦察 reconnaissance)"}


def meta(path):
    """dv -> (wall seconds, device)."""
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 7:
            out[float(p[1])] = (float(p[5]), " ".join(p[6:]))
    return out


def section(path):
    b, vals = read(path)
    vals = geo_tail(vals)
    if len(vals) < 3:
        return
    m = meta(path)
    T = romberg(vals)
    val, err, col = best_column(T)
    band = 10.0 * err
    qmax = max(2, int((1.0 / (2.0 * band)) ** 0.5))
    cs = candidates(val, band, qmax)
    print(f"\n### b = {b} — {NAME[b]}  ·  论文当前值 {PAPER[b]}\n")
    print("| dv | 网格 grid | 阶梯原值 raw ladder value | Richardson j=1 "
          "| j=2 | j=3 | 墙钟 wall | 设备 device |")
    print("|---|---|---|---|---|---|---|---|")
    for k, (dv, v) in enumerate(vals):
        n = int(round(4 / dv))
        sec, dev = m.get(dv, (0.0, "-"))
        c = [f"`{T[k][j]:+.12f}`" if j < len(T[k]) else "" for j in (1, 2, 3)]
        print(f"| {dv:g} | {n}^{b-1} | `{v:+.15f}` | {c[0]} | {c[1]} | "
              f"{c[2]} | {sec:.1f} s | {dev} |")
    print(f"\n最优列 best column **j = {col}**，相邻行漂移 drift = {err:.1e} "
          f"→ 误差带 band = {band:.1e}，唯一性分母上限 qmax = {qmax}\n")
    print(f"**外推值 extrapolated = {val:+.15f}**\n")
    for f, d in cs[:4]:
        print(f"- 候选 candidate `{f}` = {float(f):+.15f}, |dev| = {d:.2e}")
    if len(cs) == 1:
        print(f"\n> ### ⇒ 唯一有理重构 UNIQUE RECONSTRUCTION: **{cs[0][0]}**")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "results"
    for b in (3, 4, 5, 6, 7):
        p = os.path.join(d, f"res_b{b}.txt")
        if os.path.exists(p):
            section(p)
