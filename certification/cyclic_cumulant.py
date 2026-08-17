# -*- coding: utf-8 -*-
"""F-CYC gate: partition-cyclic cumulant formula for the connected
integrands (determinantal process), paper §5.5(ii).
F-CYC 门: 连通被积函数的分拆-循环累积量公式 (行列式过程), 论文 §5.5(ii).

Formula / 公式:
  C_b(v₁..v_b) = Σ_{P ⊢ [b]} (−1)^{|P|−1} Σ_{σ: 块的循环序}
                 ov(0, w_{σ1}, w_{σ1}+w_{σ2}, ...),
  w_i = 块内频率和.  b=2 实例: 1 − (1−|v|)₊ = min(|v|,1) ✓.
  项数 Σ_m S(b,m)(m−1)!: b=7 为 9366 (vs Fubini 47293).

Gate criterion / 判据: matches the direct Ursell engine C_of at
b=3,4,5 (random points + grid slices) to ≤ 1e-12.
用法: python3 cyclic_cumulant.py {check|c7 DV [i0 i1]}
"""
import itertools
import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from cumulant_engine import C_of, overlap


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def cyc_terms(b):
    """预生成: (符号, 块索引序) 列表 — 每分拆 × 每循环序.
    循环序规范化: 含元素 0 的块固定在首位, 其余 (m−1)! 排列."""
    terms = []
    for P in set_partitions(list(range(b))):
        m = len(P)
        sign = (-1.0) ** (m - 1)
        first = next(i for i, blk in enumerate(P) if 0 in blk)
        others = [i for i in range(m) if i != first]
        for perm in itertools.permutations(others):
            order = [first] + list(perm)
            terms.append((sign, [P[i] for i in order]))
    return terms


def cyc_C(vlist, terms):
    """F-CYC 求值: vlist = b 个同形数组 (Σv = 0)."""
    tot = np.zeros_like(vlist[0])
    for sign, blocks in terms:
        ws = [sum(vlist[i] for i in blk) for blk in blocks]
        pos = [np.zeros_like(vlist[0])]
        acc = np.zeros_like(vlist[0])
        for w in ws[:-1]:
            acc = acc + w
            pos.append(acc.copy())
        tot = tot + sign * overlap(pos)
    return tot


def mode_check():
    rng = np.random.default_rng(3)
    ok_all = True
    for b in (3, 4, 5):
        terms = cyc_terms(b)
        vs = [rng.uniform(-2, 2, 4000) for _ in range(b - 1)]
        vs.append(-sum(vs))
        ref = C_of(vs)
        new = cyc_C(vs, terms)
        dev = float(np.abs(ref - new).max())
        ok = dev < 1e-12
        ok_all &= ok
        print(f"  b={b}: 项数 {len(terms)}, 最大偏差 {dev:.2e} "
              f"{'✓' if ok else '✗ 鸣响'}")
    print(f"[F-CYC] {'PASS' if ok_all else 'FIRE'}")


def mode_c7(dv, i0=0, i1=None):
    """C₇ = ∫O₇C₇ d⁶v 经 F-CYC (9366 项), A-分片检查点."""
    terms = cyc_terms(7)
    g = np.arange(-2, 2 + dv / 2, dv)
    n = len(g)
    i1 = n if i1 is None else min(i1, n)
    tag = f"/tmp/cyc_c7_dv{dv}.npz"
    try:
        d = np.load(tag)
        acc, done = float(d['acc']), int(d['done'])
    except Exception:
        acc, done = 0.0, 0
    if done > i0:
        i0 = done
    t0 = time.time()
    for i in range(i0, i1):
        A = g[i]
        B, C, D, E, F = np.meshgrid(g, g, g, g, g, indexing='ij')
        Af = np.full_like(B, A)
        vlist = [Af, B, C, D, E, F, -Af - B - C - D - E - F]
        c7 = cyc_C(vlist, terms)
        ov = overlap([np.zeros_like(B), Af, Af + B, Af + B + C,
                      Af + B + C + D, Af + B + C + D + E,
                      Af + B + C + D + E + F])
        acc += float((ov * c7).sum()) * dv ** 6
        np.savez(tag, acc=acc, done=i + 1)
        if time.time() - t0 > 480:
            print(f"[c7] dv={dv}: 检查点 @ {i+1}/{n}, "
                  f"部分和 {acc:+.6f}", flush=True)
            return
    print(f"[c7] dv={dv}: 完成 {i1}/{n}, ∫O₇C₇ = {acc:+.6f} "
          f"({time.time()-t0:.0f}s)")


def cyc_terms_masks(b):
    """预编译: 每项 = (符号, [走步位置的并集位掩码列表]).
    位置 k = 前 k 块并集的子集和 ⟹ 纯掩码查表 (subset-mask lookup)."""
    out = []
    for P in set_partitions(list(range(b))):
        m = len(P)
        sign = (-1.0) ** (m - 1)
        first = next(i for i, blk in enumerate(P) if 0 in blk)
        others = [i for i in range(m) if i != first]
        for perm in itertools.permutations(others):
            order = [first] + list(perm)
            masks = [0]
            cur = 0
            for bi in order[:-1]:
                for e in P[bi]:
                    cur |= (1 << e)
                masks.append(cur)
            out.append((sign, masks))
    return out


def mode_c7fast(dv, i0=0, i1=None, mid=True):
    """C₇ 提速版: 子集和缓存 (float32) + 掩码查表; mid=中点网格
    (端点网格在 b≥6 粗档符号翻转 — 论文 D10; 中点网格稳定)."""
    terms = cyc_terms_masks(7)
    g = (np.arange(-2 + dv / 2, 2, dv) if mid
         else np.arange(-2, 2 + dv / 2, dv))
    n = len(g)
    i1 = n if i1 is None else min(i1, n)
    tag = f"/tmp/cyc_c7f_dv{dv}_m{int(mid)}.npz"
    try:
        d = np.load(tag)
        acc, done = float(d['acc']), int(d['done'])
    except Exception:
        acc, done = 0.0, 0
    if done > i0:
        i0 = done
    t0 = time.time()
    for i in range(i0, i1):
        A = g[i]
        B, C, D, E, F = np.meshgrid(g, g, g, g, g, indexing='ij')
        B = B.astype(np.float32)
        vs = [np.full_like(B, np.float32(A)), B,
              C.astype(np.float32), D.astype(np.float32),
              E.astype(np.float32), F.astype(np.float32)]
        vs.append(-sum(vs))
        # 子集和缓存 (全 128 掩码, 升序保证前驱存在)
        SS = {0: np.zeros_like(B)}
        for mk in range(1, 128):
            lo = mk & (-mk)
            SS[mk] = SS[mk ^ lo] + vs[lo.bit_length() - 1]
        c7 = np.zeros_like(B)
        for sign, masks in terms:
            arrs = [SS[mk] for mk in masks]
            mx = np.maximum.reduce(arrs)
            mn = np.minimum.reduce(arrs)
            c7 += np.float32(sign) * np.clip(1.0 - (mx - mn), 0.0,
                                             None)
        # 外 O₇ 走步
        w = [np.zeros_like(B)]
        s = np.zeros_like(B)
        for v in vs[:-1]:
            s = s + v
            w.append(s.copy())
        ov = np.clip(1.0 - (np.maximum.reduce(w)
                            - np.minimum.reduce(w)), 0.0, None)
        acc += float((ov.astype(np.float64)
                      * c7.astype(np.float64)).sum()) * dv ** 6
        np.savez(tag, acc=acc, done=i + 1)
        if time.time() - t0 > 480:
            print(f"[c7f] dv={dv}: 检查点 @ {i+1}/{n}, "
                  f"部分和 {acc:+.6f}", flush=True)
            return
    print(f"[c7f] dv={dv}: 完成 {i1}/{n}, ∫O₇C₇ = {acc:+.6f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    if sys.argv[1] == 'check':
        mode_check()
    elif sys.argv[1] == 'c7fast':
        dv = float(sys.argv[2])
        i0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        i1 = int(sys.argv[4]) if len(sys.argv) > 4 else None
        mode_c7fast(dv, i0, i1)
    else:
        dv = float(sys.argv[2])
        i0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        i1 = int(sys.argv[4]) if len(sys.argv) > 4 else None
        mode_c7(dv, i0, i1)
