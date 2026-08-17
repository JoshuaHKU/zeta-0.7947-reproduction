# -*- coding: utf-8 -*-
"""Tail bound via partial-sum identity + Abel + V(θ), tensor-sieve
version (helper for the §4 chain and the quadrature constant).
尾界: 部分和恒等式 + Abel + V(θ), 张量筛版 (§4 链与积分常数辅助).

部分和层换序合法 ({M/d} 权 ℓ¹):
    Σ_{j≤M}(𝔖−D̄)(j) = −Σ_d d·{M/d}·γ^free_d        (F-E3d)
    |Σ_j tail_P·row| ≤ SB(Jmax)·V(θ),  SB(M) = Σ_d d·min(1,M/d)|γ^{>P}_d|

γ^free_d 的带符号筛法数组: generic-支撑基座 (奇平方自由, 系数
(p−1)⁻²/(1−(p−1)⁻²)) ⊗ 特殊素数 (p | 2b₁b₂) 的 (v, f_v)-权层;
类-d 上以慢路径精确替换 (γ_full − γ_class).

预注册证伪器:
  F-E3d  恒等式 (张量筛 D=2e5): (2,3)/(5,5), M ∈ {100, 1000}:
         |差| ≤ 0.01·max(|直接|,1) + 余项界(2×[1e5,2e5]-段×M).
  F-E3c  包络 env_total(T=2400, P=40, DCAP=2e4, 含 d>DCAP 余项):
         ≤ 0.02 → E3 闭合; (0.02,0.1] 部分鸣响; > 0.1 结构鸣响.

用法: python3 tail_bound.py ident | env 2400
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from m1_suite import (C2_TWIN, Sieves, dbar_alpha, primepowers,
                      prime_factors, vp_int)
from mains_envelope import _beta_loc, beta_q, class_split, mu

TWO_PI = 2 * math.pi
LOG2 = math.log(2.0)


def primes_list(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.nonzero(s)[0]


def _loc(p, v, e1, e2):
    """局部 γ-因子 f_v = β_p(v) − β_p(v+1)."""
    return (_beta_loc(p, v, e1, e2) - _beta_loc(p, v + 1, e1, e2))


def gamma_signed_array(D, b1, b2):
    """带符号 γ^free_d 数组 (d ≤ D), 张量筛."""
    S = sorted(prime_factors(2 * b1 * b2))          # 含 2
    S_odd = [p for p in S if p > 2]
    # generic 基座: ∏_{generic 奇 p}(1−(p−1)⁻²) = C₂ ÷ 特殊奇素因子
    base0 = C2_TWIN
    for p in S_odd:
        base0 /= (1 - 1.0 / (p - 1) ** 2)
    arr = np.full(D + 1, base0)
    arr[0] = 0.0
    for p in primes_list(D):
        p = int(p)
        if p in S:
            continue
        r = (1.0 / (p - 1) ** 2) / (1 - 1.0 / (p - 1) ** 2)
        arr[p::p] *= r
        if p * p <= D:
            arr[p * p::p * p] = 0.0
    # 零掉特殊素数的倍数 (generic-only 支撑)
    for p in S:
        arr[p::p] = 0.0
    # 逐特殊素数张量: new[p^v·m] = f_v · old[m] (p ∤ m; v 从 0 起)
    for p in S:
        e1, e2 = vp_int(b1, p), vp_int(b2, p)
        new = np.zeros(D + 1)
        v, pv = 0, 1
        while pv <= D:
            fv = _loc(p, v, e1, e2)
            if fv != 0.0:
                mm = np.arange(1, D // pv + 1)
                keep = mm % p != 0
                new[pv * mm[keep]] += fv * arr[mm[keep]]
            v += 1
            pv *= p
        arr = new
    # 类-d 精确替换 (γ_full − γ_class)
    cls = [1]
    for p in S:
        ext = []
        for c in cls:
            pv = c * p
            while pv <= D:
                ext.append(pv)
                pv *= p
        cls += ext
    for d in sorted(set(cls)):
        arr[d] = _gamma_free_exact(d, b1, b2)
    return arr


def _gamma_free_exact(d, b1, b2):
    """逐 d 精确 γ^free_d (慢路径; 类-d 与小 d 校验用).

    修正 (前向记录): 特殊奇素数的通用因子 (1−(p−1)⁻²) 的移除与 p|d 无关 —
    首版只在 p∤d 时移除, 已被张量筛失配定位 (比值恰 (1−(p−1)⁻²)⁻¹).
    """
    ps_d = prime_factors(d) if d > 1 else set()
    ps_b = prime_factors(2 * b1 * b2)
    val = 1.0
    for p in ps_d:
        e1, e2 = vp_int(b1, p), vp_int(b2, p)
        val *= _loc(p, vp_int(d, p), e1, e2)
    corr = 1.0
    for p in ps_b:
        if p in ps_d:
            continue
        e1, e2 = vp_int(b1, p), vp_int(b2, p)
        corr *= _loc(p, 0, e1, e2)
    tail = C2_TWIN
    for p in ps_b:
        if p > 2:
            tail /= (1 - 1.0 / (p - 1) ** 2)
    for p in ps_d:
        if p > 2 and p not in ps_b:
            tail /= (1 - 1.0 / (p - 1) ** 2)
    g_full = val * corr * tail
    if all(p in ps_b for p in ps_d):
        vc = 1.0
        for p in ps_d:
            e1, e2 = vp_int(b1, p), vp_int(b2, p)
            vc *= _loc(p, vp_int(d, p), e1, e2)
        for p in ps_b:
            if p not in ps_d:
                e1, e2 = vp_int(b1, p), vp_int(b2, p)
                vc *= _loc(p, 0, e1, e2)
        return g_full - vc
    return g_full


def ident_check():
    """F-E3d: 张量筛 D=2e5, 全 d-范围带符号求和 + 余项界."""
    sv = Sieves(20000)
    D = 200000
    ok_all = True
    for b1, b2 in [(2, 3), (5, 5)]:
        sarr = sv.s_array(b1, b2)
        db = np.zeros(sv.J + 1)
        for c, al in dbar_alpha(b1, b2):
            db[c::c] += al
        g = gamma_signed_array(D, b1, b2)
        # 抽查张量筛 vs 慢路径 (20 个随机 d)
        rng = np.random.default_rng(7)
        bad = 0
        for d in rng.integers(1, D, 20):
            if abs(g[d] - _gamma_free_exact(int(d), b1, b2)) > 1e-12:
                bad += 1
        print(f"  ({b1},{b2}) 张量筛 vs 慢路径 抽查: {20-bad}/20")
        dd = np.arange(D + 1, dtype=np.float64)
        rem_seg = float(np.sum(np.abs(g[D // 2:])))
        for M in [100, 1000]:
            lhs = float(np.sum(sarr[1:M + 1] - db[1:M + 1]))
            frac = (M / np.maximum(dd, 1)) % 1.0
            rhs = -float(np.sum(dd * frac * g))
            rem = 2.0 * M * rem_seg
            ok = abs(lhs - rhs) <= 0.01 * max(abs(lhs), 1.0) + rem
            ok_all &= ok
            print(f"  ({b1},{b2}) M={M:>5}: 直接={lhs:+.5f} "
                  f"γ-形={rhs:+.5f} |差|={abs(lhs-rhs):.5f} "
                  f"余项界={rem:.5f} {'✓' if ok else '✗'}")
    return ok_all


def env_total(T, P=40, DCAP=20000):
    """尾包络 zone-积分 (F-E3c), 含 d > DCAP 余项."""
    X = int(T / TWO_PI)
    l = math.log(T / TWO_PI)
    ell1 = l + 2 * LOG2 - 1
    norm4 = l * X * ell1 ** 4
    thmin = 4 * math.pi ** 2 / T
    th = np.exp(np.linspace(math.log(thmin), math.log(200.0),
                            int(math.log(200.0 / thmin) * 320) + 2))
    om = np.log(TWO_PI / th)
    V = np.zeros(len(th))
    for i, t in enumerate(th):
        J = min(int(200 / t) + 1, 60000)
        j = np.arange(1, J + 2)
        row = (np.sin(2 * t * j) - np.sin(t * j)) / j
        V[i] = float(np.sum(np.abs(np.diff(row)))) + abs(row[0]) \
            + 4.0 / max(J, 1)
    pp, lw = primepowers(X)

    def qvec(be1, be2):
        m1_, m2_ = l + om - be1, l + om - be2

        def o4v(a, b, c):
            z = np.zeros_like(a)
            mx = np.maximum.reduce([z, a, b, c])
            mn = np.minimum.reduce([z, a, b, c])
            return np.clip(l - (mx - mn), 0, None)

        tot = np.zeros_like(om)
        for u1, u3 in ((be1 + 0 * om, m1_), (m1_, be1 + 0 * om)):
            for u2, u4 in ((be2 + 0 * om, m2_), (m2_, be2 + 0 * om)):
                de = u1 + u3 - u2 - u4
                tot += (o4v(-u1 + de, u3 - u4, -u4)
                        + o4v(u2 - u3 - u4, -u3 - u4, -u4)
                        + o4v(-u2 - u3 + u4, -u3 + u4, u4))
        return tot

    Mgrid = np.exp(np.linspace(math.log(2.0), math.log(60000.0), 25))
    ds = None
    tot_env = 0.0
    t_last = time.time()
    npair = 0
    for i in range(len(pp)):
        for k in range(i, len(pp)):
            b1, b2 = int(pp[i]), int(pp[k])
            w = lw[i] * lw[k] / (b1 * b2)
            sel = th <= T / max(b1, b2) ** 2
            sel &= (min(b1, b2) >= TWO_PI / th) | (th >= TWO_PI)
            if not sel.any():
                continue
            ths = th[sel]
            q = qvec(math.log(b1), math.log(b2))[sel]
            if b1 != b2:
                q = q + qvec(math.log(b2), math.log(b1))[sel]
            if not q.any():
                continue
            npair += 1
            g = gamma_signed_array(DCAP, b1, b2)
            ag = np.abs(g)
            # >P 排除: d ≤ P 的 γ^{>P} 精确
            for d in range(1, P + 1):
                gg = _gamma_free_exact(d, b1, b2)
                for e in range(1, P // d + 1):
                    de = d * e
                    if mu(e) != 0 and not class_split(de, b1, b2):
                        gg -= mu(e) * beta_q(de, b1, b2)
                ag[d] = abs(gg)
            if ds is None or len(ds) != DCAP + 1:
                ds = np.arange(DCAP + 1, dtype=np.float64)
            SBg = np.array([float(np.sum(
                ds * np.minimum(1.0, M / np.maximum(ds, 1.0)) * ag))
                for M in Mgrid])
            rem_half = float(np.sum(ag[DCAP // 2:]))
            SBg = SBg + Mgrid * 2.0 * rem_half
            Jm = np.minimum(200.0 / ths, 60000.0)
            SB = np.interp(np.log(Jm), np.log(Mgrid), SBg)
            tot_env += w * T * np.trapezoid(
                SB * V[sel] * np.abs(q) / ths ** 2, ths)
    return tot_env / math.pi / norm4, npair


def main():
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'ident'
    if mode == 'ident':
        print("[F-E3d] 部分和恒等式 (张量筛 D=2e5):")
        ident_check()
    elif mode == 'env':
        T = float(sys.argv[2])
        ev, np_ = env_total(T)
        verdict = ('✓ E3 闭合' if ev <= 0.02 else
                   ('部分鸣响' if ev <= 0.1 else '✗ 结构鸣响'))
        print(f"[F-E3c] T={T:.0f}, P=40, DCAP=2e4: env_total = {ev:.5f}"
              f"  ({np_} 对)  → {verdict}")
    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
