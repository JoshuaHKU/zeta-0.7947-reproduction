# -*- coding: utf-8 -*-
"""Reference implementation of the {4,2} spectator-pair variant
(paper §5.5; the one connected constant still banded).  This file
IS the specification for the CUDA port: same class structure as
the archived assembly runner, midpoint protocol.

{4,2} 旁观对变体的参考实现（论文 §5.5；唯一仍带宽的连通常数）。
本文件即 CUDA 移植的规格：类结构与存档装配 runner 相同，中点协议。

DEFINITION / 定义
-----------------
Free variables (4D): v (pair frequency), c1, c2, c3 (4-block free
frequencies; the 4th block frequency is c4 = −c1−c2−c3).  Midpoint
grid per dimension: g = arange(−2 + dv/2, 2, dv).

Integrand = ovV(walk) · C4(c1,c2,c3,c4) · C2(v), where
  C2(v)  = min(|v|, 1);
  C4     = the 4-point sine-model Ursell cumulant, evaluated as the
           26-term partition-cyclic signed overlap sum (identical to
           compile_terms(4) of the pure-cycle engine, positions
           Σ_{j∈mask} w_j over w = (c1,c2,c3,c4));
  ovV    = (1 − (max p − min p))₊ over the 6-point prefix walk p of
           the full cycle, with the pair frequencies (v, −v) placed
           at cycle distance d and the block frequencies in order:

  d=1 (pair at slots {1,2}):  p = [0, v, 0,    c1,      c1+c2, c1+c2+c3]
  d=2 (pair at slots {1,3}):  p = [0, v, v+c1, c1,      c1+c2, c1+c2+c3]
  d=3 (pair at slots {1,4}):  p = [0, v, v+c1, v+c1+c2, c1+c2, c1+c2+c3]

Class values U_d = Σ_grid ovV·C4·C2 · dv⁴, and the constant is

  {4,2} = 6·U₁ + 6·U₂ + 3·U₃        (multiplicities 6/6/3).

Global negation symmetry: the integrand is invariant under
(v,c1,c2,c3) → −(v,c1,c2,c3) (walk reflects, C2/C4 even), so the
--sym halving over the v-slices applies exactly as in the pure
engine.  Support pruning: ovV vanishes unless the 6-walk span ≤ 1
(same pruning as b=6); C4's own overlap factors prune each of its
26 terms identically.

CALIBRATION GATES for the CUDA port / CUDA 移植校准门
-----------------------------------------------------
G1 (C4 mechanics): the same 26-term evaluator on the pure 4-cycle
   walk must reproduce the b=4 ladder of the pure engine
   (−0.016874375 at dv=0.05, −1/60 in the limit).
G2 (bitwise vs this reference): U₁,U₂,U₃ at dv=0.1 and 0.05 must
   match the values recorded below to ≤1e-14 (fp64 order effects).
G3 (band): the final ladder must land inside −0.0552(8), and the
   archived midpoint reads −0.0588 (dv=0.1) / −0.0558 (dv=0.05)
   are reproduced by this reference (they are gates G2).
G4 (identification): Romberg + rational reconstruction with the
   uniqueness qmax.  PRE-REGISTERED CANDIDATE (F-RAT-42, from the
   three reference rungs below): {4,2} = −23/420 = −0.0547619…
   — minimal-denominator reconstruction of the h² extrapolation
   −0.0547626, error-ratio profile 3.99 ≈ 4 against it, inside
   the band, and in the denominator family of {2,2,2} = 131/420.
   The V100 ladder decides; deviation of the fine rungs from the
   h² prediction e(dv) = c·dv² (c ≈ 0.407) fires the gate.

REFERENCE VALUES (this file, midpoint, numpy; gates G2) /
参考值（G2 门，逐位对照）:
   dv=0.1  : U1=-0.006870600  U2=-0.000765600  U3=-0.004327950
             {4,2} = -0.058801050
   dv=0.05 : U1=-0.006480752  U2=-0.000786689  U3=-0.004058445
             {4,2} = -0.055779977
   dv=0.025: U1=-0.006382168  U2=-0.000791914  U3=-0.003990818
             {4,2} = -0.055016941
   (archive cross-check: r107 midpoint reads -0.0588 / -0.0558 at
   dv=0.1 / 0.05 — reproduced exactly.)

LADDER SCHEDULE (V100) / 阶梯排程:
  dv = 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625
  (4D; with support pruning and --sym well under an hour total).

REPORT BACK / 回传: per dv: (U₁, U₂, U₃, total, wall, device);
then the Romberg table and reconstruction as in
COMPUTATION_REPORT.md.

Usage: python3 spectator_42_reference.py DV [i0 i1]   (v-slice
checkpointed, chunked; numpy reference — slow but exact spec).
"""
import itertools
import sys
import time

import numpy as np


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def compile_terms(b):
    """(sign, prefix-mask) family for C_b — identical to the pure
    engine.  掩码族与纯圈引擎一致."""
    out = []
    for P in set_partitions(list(range(b))):
        m = len(P)
        sign = (-1.0) ** (m - 1)
        first = next(i for i, blk in enumerate(P) if 0 in blk)
        others = [i for i in range(m) if i != first]
        for perm in itertools.permutations(others):
            order = [first] + list(perm)
            masks, cur = [], frozenset()
            for bi in order[:-1]:
                cur = cur | frozenset(P[bi])
                masks.append(tuple(sorted(cur)))
            out.append((sign, masks))
    return out


TERMS4 = compile_terms(4)


def C4_of(w):
    """26-term partition-cyclic C4 at frequency list w (arrays,
    Σw = 0).  26 项分拆-循环 C₄."""
    sub = np.zeros_like(w[0])
    for sign, masks in TERMS4:
        if not masks:
            sub = sub + sign
            continue
        mx = np.zeros_like(w[0])
        mn = np.zeros_like(w[0])
        for mk in masks:
            pos = sum(w[j] for j in mk)
            mx = np.maximum(mx, pos)
            mn = np.minimum(mn, pos)
        sub = sub + sign * np.clip(1.0 - (mx - mn), 0.0, None)
    return sub


def ov(walk):
    mx = np.maximum.reduce(walk)
    mn = np.minimum.reduce(walk)
    return np.clip(1.0 - (mx - mn), 0.0, None)


def run(dv, i0=0, i1=None):
    g = np.arange(-2 + dv / 2, 2, dv)
    n = len(g)
    i1 = n if i1 is None else min(i1, n)
    tag = f"/tmp/spec42_dv{dv}.npz"
    try:
        d = np.load(tag)
        acc, done = d["acc"], int(d["done"])
    except Exception:
        acc, done = np.zeros(3), 0
    if done > i0:
        i0 = done
    t0 = time.time()
    for i in range(i0, i1):
        v = g[i]
        C1, C2_, C3 = np.meshgrid(g, g, g, indexing="ij")
        c4 = C4_of([C1, C2_, C3, -C1 - C2_ - C3])
        dd = c4 * min(abs(v), 1.0)
        Z = np.zeros_like(C1)
        V = np.full_like(C1, v)
        acc[0] += float((ov([Z, V, Z, C1, C1 + C2_, C1 + C2_ + C3])
                         * dd).sum()) * dv ** 4
        acc[1] += float((ov([Z, V, V + C1, C1, C1 + C2_,
                             C1 + C2_ + C3]) * dd).sum()) * dv ** 4
        acc[2] += float((ov([Z, V, V + C1, V + C1 + C2_, C1 + C2_,
                             C1 + C2_ + C3]) * dd).sum()) * dv ** 4
        np.savez(tag, acc=acc, done=i + 1)
        if time.time() - t0 > 480:
            print(f"[42-ref] dv={dv}: checkpoint {i+1}/{n} "
                  f"U={acc}", flush=True)
            return
    u1, u2, u3 = acc
    print(f"[42-ref] dv={dv}: U1={u1:+.9f} U2={u2:+.9f} "
          f"U3={u3:+.9f}  {{4,2}} = 6U1+6U2+3U3 = "
          f"{6*u1+6*u2+3*u3:+.9f}  ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    dv = float(sys.argv[1])
    i0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    i1 = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run(dv, i0, i1)
