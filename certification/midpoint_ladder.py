# -*- coding: utf-8 -*-
"""Grouped midpoint-ladder engine for the connected constants,
paper §5.5: ∫O_bC_b = Σ_{(P,σ)} sign·[nonnegative ov·ov integral].
Each term is nonnegative and smooth, so midpoint grids converge
monotonically (oscillation is isolated in the exact ± signs between
terms). Calibration gate: b=4 vs Φ₄ = −1/60 exact. The ladders give
C₅ = 0.0278(1) and {6} = −0.0078(8) (endpoint-grid evaluations are
excluded by the sign-flip pathology, paper D10).

连通常数的分组中点阶梯引擎 (论文 §5.5): 每个 (P,σ)-项非负光滑 ⟹
中点网格单调收敛 (振荡被隔离到项间精确 ± 号). b=4 对 Φ₄ = −1/60
精确校准; 阶梯给出 C₅ = 0.0278(1), {6} = −0.0078(8) (端点网格因
符号翻转病理被排除, 论文 D10).

用法: python3 midpoint_ladder.py B DV [i0 i1]   (v₁-分片检查点)
"""
import itertools
import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from cyclic_cumulant import set_partitions
from cumulant_engine import overlap


def build_terms(b):
    """(sign, 并块前缀的元素集列表) — 预编译."""
    out = []
    for P in set_partitions(list(range(b))):
        m = len(P)
        sign = (-1.0) ** (m - 1)
        first = next(i for i, blk in enumerate(P) if 0 in blk)
        others = [i for i in range(m) if i != first]
        for perm in itertools.permutations(others):
            order = [first] + list(perm)
            # 前缀并集 (掩码), 不含全集 (最后位置省略 — walk 只到 m−1)
            masks = []
            cur = frozenset()
            for bi in order[:-1]:
                cur = cur | frozenset(P[bi])
                masks.append(sorted(cur))
            out.append((sign, masks))
    return out


def run(b, dv, i0=0, i1=None):
    terms = build_terms(b)
    g = np.arange(-2 + dv / 2, 2, dv)      # midpoint grid / 中点网格 (论文 §5.5)
    n = len(g)
    i1 = n if i1 is None else min(i1, n)
    tag = f"/tmp/mladder_b{b}_dv{dv}.npz"
    try:
        d = np.load(tag)
        acc, done = float(d['acc']), int(d['done'])
    except Exception:
        acc, done = 0.0, 0
    if done > i0:
        i0 = done
    t0 = time.time()
    for i in range(i0, i1):
        v1 = g[i]
        mesh = np.meshgrid(*([g] * (b - 2)), indexing='ij')
        vs = [np.full_like(mesh[0], v1)] + list(mesh)
        vs.append(-sum(vs))
        # 外 O_b: v-前缀走步
        walk = [np.zeros_like(vs[0])]
        s = np.zeros_like(vs[0])
        for v in vs[:-1]:
            s = s + v
            walk.append(s.copy())
        ovV = overlap(walk)
        sub = np.zeros_like(vs[0])
        for sign, masks in terms:
            pos = [np.zeros_like(vs[0])]
            for mk in masks:
                pos.append(sum(vs[j] for j in mk))
            sub = sub + np.float64(sign) * overlap(pos)
        acc += float((ovV * sub).sum()) * dv ** (b - 1)
        np.savez(tag, acc=acc, done=i + 1)
        if time.time() - t0 > 480:
            print(f"[b={b}] dv={dv}: 检查点 @ {i+1}/{n}, "
                  f"部分和 {acc:+.6f}", flush=True)
            return
    print(f"[b={b}] dv={dv}: 完成, ∫O_bC_b = {acc:+.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    b = int(sys.argv[1])
    dv = float(sys.argv[2])
    i0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    i1 = int(sys.argv[4]) if len(sys.argv) > 4 else None
    run(b, dv, i0, i1)
