# -*- coding: utf-8 -*-
"""Quantitative band ledger (helper for the sawtooth chain, paper
§4): mid-band cell ledger, decoupling statistics, cell-mean anchor.
带定量账本 (锯齿链辅助模块, 论文 §4): 中段带胞元账本、解耦统计、
胞元均值锚. 三种模式:

  band     中段带胞元账本 (K-加权分裂): F-A0..F-A3
  g2       T2 演示族黏合权统计与解耦协方差 (F-G2)
  cellmean 胞元级光滑均值 B_cellmean(C=40) 对照实测 (F-C1) + LP 重跑

带定义 (自含):
    BAND = { X < min(N₁,N₂) ≤ T^{3/2} } ∩ { max(b₁,b₂) < T³/N̄² }
    窗口 |N₁−N₂|·T ≤ 40·min (C=40);  b_i = 各积最小腿;  m,n ≤ X 强制.
    胞元 = (b₁,b₂,j,块);  块 = min-侧 N 几何分块 ×1.3.

K-加权分裂 (本轮方法): 核吸进胞元质量, 几何权只留慢变部分 —
    Ã(cell)  = Σ_pairs Λ(m)Λ(n)·K(Δ_pair)   (K = Ŵin, Δ 逐对精确)
    MT̃(cell) = 𝔖(j)·∫_block K(Δ(N,j)) dN /(b₁b₂)
    w̃(cell)  = (1/π d ℓ₁⁴)·Λ(b₁)Λ(b₂)·Q̄sym(块中心)/N̄
    R_band   = Σ w̃·Ã + (Q,1/N 慢变价格, ≤ (√1.3−1) 相对)
    D̃ = Σ(Ã−MT̃)²  是 K-加权条带计数的色散 (定理侧: BDH/Mikawa
    容许光滑权; 见备忘录). CS: |Σw̃(Ã−MT̃)| ≤ √(Σw̃²)·√(ΣD̃) [全局].

预注册证伪器 (计算前固定):
  F-A0  分裂重构: |Σ w̃·Ã − R_band^直接| ≤ 0.20·Σ|w̃·Ã|.
  F-A1  CS-账本 (T=4800): L_CS = √(Σw̃²)·√(ΣD̃) ≤ 0.0383 → GO;
        ≤ 2× → 部分鸣响 (需分组精化); > 2× → 结构性鸣响.
  F-A2  定理地板: L_floor = √(Σw̃²)·√(Σdiag̃), diag̃ = Σ Λ²Λ²K² (精确).
  F-A3  定理链代理: L_thm = √(Σw̃²)·√(Σ[diag̃ + C_sel·|D̃−diag̃|]),
        C_sel = 4 (可引用孪生 Selberg 常数; 定理不能用 B_net 的符号).
  F-G2  解耦: |Σ C_dec| ≤ 0.25 × 离对角质量.
  F-C1  均值锚: B_cellmean(C=40) 对实测 (.0047/.0039/.0031/.0023)
        逐高度 ≤ 30%, 残差随 l 不增.

用法: python3 g1_ledger.py band 2400 | band 4800 | g2 | cellmean
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from m1_suite import Sieves, primepowers
from babs_mean import leg_pairs, o4_block

TWO_PI = 2 * math.pi
LOG2 = math.log(2.0)
C_TAPER = 40.0
C_SEL = 4.0


def qsym_exact(l, b1, m1, b2, m2):
    """O₄ 腿序和, 多重数与管线枚举严格一致 (平方积单腿序)."""
    def vol(a, b, c):
        mx = max(0.0, a, b, c)
        mn = min(0.0, a, b, c)
        return max(l - (mx - mn), 0.0)

    def o4(u1, u3, u2, u4):
        de = u1 + u3 - u2 - u4
        return (vol(-u1 + de, u3 - u4, -u4)
                + vol(u2 - u3 - u4, -u3 - u4, -u4)
                + vol(-u2 - u3 + u4, -u3 + u4, u4))

    s1 = [(b1, m1)] if abs(b1 - m1) < 1e-12 else [(b1, m1), (m1, b1)]
    s2 = [(b2, m2)] if abs(b2 - m2) < 1e-12 else [(b2, m2), (m2, b2)]
    return sum(o4(math.log(a1), math.log(a3), math.log(a2), math.log(a4))
               for a1, a3 in s1 for a2, a4 in s2)


# ----------------------------------------------------------------------
# 带账本 (band 模式)
# ----------------------------------------------------------------------
def band_ledger(T):
    X = int(T / TWO_PI)
    l = math.log(T / TWO_PI)
    ell1 = l + 2 * LOG2 - 1
    norm = 1.0 / math.pi / (l * X * ell1 ** 4)
    Nmax = T ** 1.5
    edges = [float(X)]
    while edges[-1] < Nmax:
        edges.append(min(edges[-1] * 1.3, Nmax))
    nblk = len(edges) - 1
    edges = np.array(edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])          # 块中心
    pp, lw = primepowers(X)
    lam = np.zeros(X + 2)
    lam[pp] = lw
    lamd = {int(p): float(w) for p, w in zip(pp, lw)}
    bmax = int(T ** 0.75) + 1
    bs = [int(b) for b in pp if b <= bmax]
    sv = Sieves(60000)
    JOFF = 60000

    tot = dict(D=0.0, W2=0.0, wA=0.0, wMT=0.0, abswA=0.0, diag=0.0,
               Dthm=0.0, ncell=0,
               # 粒度阶梯: j-求和执行后的粗胞元 (b₁,b₂,块)
               Dc=0.0, W2c=0.0, ncoarse=0, abswAc=0.0)
    sarr_cache = {}
    for b1 in bs:
        for b2 in bs:
            sup_hi = min(b2 * X, b1 * X * 1.05, Nmax)
            if sup_hi <= X:
                continue
            nlo = max(int(X // b1), 2)
            nhi = min(int(sup_hi / b1) + 1, X)
            if nhi <= nlo:
                continue
            narr0 = np.arange(nlo, nhi + 1)
            lamn0 = lam[narr0]
            k0 = lamn0 > 0
            narr0, lamn0 = narr0[k0], lamn0[k0]
            if len(narr0) == 0:
                continue
            tmax = int(C_TAPER * X / T) + 2
            base = b1 * narr0
            m0 = base // b2
            KA, VA, KW = [], [], []
            for t in range(-tmax, tmax + 1):
                m = m0 + t
                jv = base - b2 * m
                N2 = b2 * m
                N1 = base
                mn = np.minimum(N1, N2)
                with np.errstate(divide='ignore'):
                    ok = ((m >= 2) & (m <= X) & (jv != 0)
                          & (np.abs(jv) * T <= C_TAPER * mn)
                          & (mn > X) & (mn <= Nmax))
                if not ok.any():
                    continue
                mm, nn = m[ok], narr0[ok]
                lm = lam[mm]
                k2 = lm > 0
                if not k2.any():
                    continue
                mm, nn, lm = mm[k2], nn[k2], lm[k2]
                jj = jv[ok][k2]
                mnv = mn[ok][k2].astype(np.float64)
                bcov = max(b1, b2) < T ** 3 / mnv ** 2
                if not bcov.any():
                    continue
                mm, nn, lm = mm[bcov], nn[bcov], lm[bcov]
                jj, mnv = jj[bcov], mnv[bcov]
                ln = lamn0[k0 * 0 + 0] if False else lam[nn]
                De = np.log((b1 * nn).astype(np.float64)
                            / (b2 * mm).astype(np.float64))
                K = (np.sin(2 * T * De) - np.sin(T * De)) / De
                blk = np.clip(np.searchsorted(edges, mnv, side='left') - 1,
                              0, nblk - 1)
                key = (jj + JOFF) * nblk + blk
                KA.append(key)
                VA.append(lm * ln * K)
                KW.append(lm * ln * K * K * lm * ln)   # Λ²Λ²K² (diag̃)
            if not KA:
                continue
            keys = np.concatenate(KA)
            vals = np.concatenate(VA)
            dvals = np.concatenate(KW)
            uk, inv = np.unique(keys, return_inverse=True)
            Acell = np.bincount(inv, weights=vals)
            Dgcell = np.bincount(inv, weights=dvals)
            jc = (uk // nblk - JOFF).astype(np.int64)
            bc = (uk % nblk).astype(np.int64)
            # MT̃: 𝔖(j)·∫_block K dN/(b₁b₂), 24 点子网格
            if (b1, b2) not in sarr_cache:
                sarr_cache[(b1, b2)] = sv.s_array(b1, b2)
                if len(sarr_cache) > 400:
                    sarr_cache.clear()
                    sarr_cache[(b1, b2)] = sv.s_array(b1, b2)
            sarr = sarr_cache[(b1, b2)]
            svals = np.where(np.abs(jc) <= sv.J,
                             sarr[np.minimum(np.abs(jc), sv.J)], 0.0)
            lo_ = np.maximum.reduce([edges[bc], np.full(len(bc), X + 1e-9),
                                     np.abs(jc) * T / C_TAPER,
                                     np.full(len(bc), 2.0 * b2)])
            hi_ = np.minimum.reduce([edges[bc + 1],
                                     np.full(len(bc), b2 * X * 1.0),
                                     b1 * X - jc.astype(np.float64)])
            MTc = np.zeros(len(uk))
            good = hi_ > lo_
            if good.any():
                gg = np.linspace(0, 1, 24)
                Ng = (lo_[good][:, None]
                      * (hi_[good] / lo_[good])[:, None] ** gg[None, :])
                Dg = np.log((Ng + jc[good][:, None]) / Ng)
                Kg = np.where(Dg != 0,
                              (np.sin(2 * T * Dg) - np.sin(T * Dg))
                              / np.where(Dg != 0, Dg, 1.0), T)
                MTc[good] = (svals[good] * np.trapezoid(Kg, Ng, axis=1)
                             / (b1 * b2))
            # w̃: 块中心几何 (Q̄, 1/N̄)
            qtab = np.array([qsym_exact(l, b1, c / b1, b2, c / b2)
                             if c / b2 >= b2 * 0 + 2 else 0.0
                             for c in ctr])
            wt = norm * lamd[b1] * lamd[b2] * qtab[bc] / ctr[bc]
            dev = Acell - MTc
            tot['D'] += float(np.sum(dev * dev))
            tot['W2'] += float(np.sum(wt * wt))
            tot['wA'] += float(np.sum(wt * Acell))
            tot['wMT'] += float(np.sum(wt * MTc))
            tot['abswA'] += float(np.sum(np.abs(wt * Acell)))
            tot['diag'] += float(np.sum(Dgcell))
            tot['Dthm'] += float(np.sum(Dgcell)
                                 + C_SEL * abs(float(np.sum(dev * dev))
                                               - float(np.sum(Dgcell))))
            tot['ncell'] += len(uk)
            # -- 粗胞元: j-求和执行 (每 (b₁,b₂,块) 一个对象) ----------
            Ac = np.bincount(bc, weights=Acell, minlength=nblk)
            Mc = np.bincount(bc, weights=MTc, minlength=nblk)
            wt_geo = norm * lamd[b1] * lamd[b2] * qtab / ctr
            occ = np.bincount(bc, minlength=nblk) > 0
            devc = (Ac - Mc)[occ]
            wg = wt_geo[occ]
            tot['Dc'] += float(np.sum(devc * devc))
            tot['W2c'] += float(np.sum(wg * wg))
            tot['abswAc'] += float(np.sum(np.abs(wg * Ac[occ])))
            tot['ncoarse'] += int(occ.sum())
    return tot


def band_direct(T, chunk=500):
    """直接四元组管线, 同带掩码: F-A0 的地面真值 (K-加权与否同一物)."""
    X = int(T / TWO_PI)
    l = math.log(T / TWO_PI)
    ell1 = l + 2 * LOG2 - 1
    norm = 1.0 / math.pi / (l * X * ell1 ** 4)
    Nmax = T ** 1.5
    na1, nb1, w1 = leg_pairs(X)
    na2, nb2, w2 = leg_pairs(X, int(Nmax) + 1)
    N1 = na1 * nb1
    N2 = na2 * nb2
    u1, u3 = np.log(na1), np.log(nb1)
    u2, u4 = np.log(na2), np.log(nb2)
    sw1 = w1 / np.sqrt(N1)
    sw2 = w2 / np.sqrt(N2)
    bl1 = np.minimum(na1, nb1).astype(np.float64)
    bl2 = np.minimum(na2, nb2).astype(np.float64)
    tot = 0.0
    for i0 in range(0, len(N1), chunk):
        s = slice(i0, i0 + chunk)
        mn = np.minimum(N1[s][:, None], N2[None, :]).astype(np.float64)
        bmx = np.maximum(bl1[s][:, None], bl2[None, :])
        mask = ((N1[s][:, None] != N2[None, :])
                & (np.abs(N1[s][:, None] - N2[None, :]) * T
                   <= C_TAPER * mn)
                & (mn > X) & (mn <= Nmax)
                & (bmx < T ** 3 / mn ** 2))
        if not mask.any():
            continue
        D = np.log(N1[s])[:, None] - np.log(N2)[None, :]
        K = np.where(mask, (np.sin(2 * T * D) - np.sin(T * D))
                     / np.where(mask, D, 1.0), 0.0)
        O4 = o4_block(l, u1[s][:, None], u3[s][:, None],
                      u2[None, :], u4[None, :])
        tot += float(np.sum(sw1[s][:, None] * sw2[None, :] * K * O4))
    return norm * tot


# ----------------------------------------------------------------------
# G2 解耦实测 (g2 模式)
# ----------------------------------------------------------------------
def g2_measure():
    from t2_swaps import Family
    print("== G2: 黏合权统计与解耦协方差 ==")
    for T in [2400., 4800.]:
        for s0 in [2.0, 3.0, 5.0]:
            fam = Family(T, 5, s0, s0 + 0.3, 2, 40, 19, m_le_X=False)
            b1, J, lam = fam.b1, fam.J, fam.lam
            off_tot, cdec_tot, vw = 0.0, 0.0, []
            for b2 in fam.b2s:
                g = math.gcd(b1, b2)
                r, q = b1 // g, b2 // g
                lo, hi = fam.m_range(b2)
                kmax = (hi - lo) // r + 1
                for k in range(-kmax, kmax + 1):
                    if k == 0:
                        continue
                    h = q * k
                    nlo = max(-((-(b2 * lo - J)) // b1), 2)
                    nhi = (b2 * hi + J) // b1
                    ns = np.arange(nlo, nhi + 1)
                    wv = np.zeros(len(ns))
                    yv = np.zeros(len(ns))
                    for i, n in enumerate(ns):
                        mlo = -((-(b1 * n - J)) // b2)
                        mhi = (b1 * n + J) // b2
                        acc = 0.0
                        for m in range(max(mlo, lo), min(mhi, hi) + 1):
                            if b1 * n - b2 * m == 0:
                                continue
                            mp = m - r * k
                            if lo <= mp <= hi and lam[m] > 0 \
                                    and lam[mp] > 0:
                                jj = b1 * (n - h) - b2 * mp
                                if jj != 0 and abs(jj) <= J:
                                    acc += lam[m] * lam[mp]
                        wv[i] = acc
                        npr = n - h
                        if 0 < npr < len(lam) and lam[n] > 0 \
                                and lam[npr] > 0:
                            yv[i] = lam[n] * lam[npr]
                    sup = wv > 0
                    off_tot += float(np.sum(wv * yv))
                    if sup.sum() < 2:
                        continue
                    wb, yb = wv[sup].mean(), yv[sup].mean()
                    cdec_tot += float(np.sum((wv[sup] - wb)
                                             * (yv[sup] - yb)))
                    vw.append(wv[sup].var() / max(wb, 1e-12) ** 2)
            ratio = abs(cdec_tot) / max(off_tot, 1e-12)
            v = float(np.mean(vw)) if vw else float('nan')
            print(f"T={T:.0f} 块[{s0}X,{s0+.3}X]: 离对角={off_tot:9.1f} "
                  f"ΣC_dec={cdec_tot:+9.1f}  比={ratio:.3f} "
                  f"{'✓' if ratio <= 0.25 else '✗ 鸣响'}  "
                  f"Var(w)/mean²={v:.2f}")


# ----------------------------------------------------------------------
# B_cell 均值锚 (cellmean 模式) + LP 重跑
# ----------------------------------------------------------------------
def cellmean():
    print("== B_cell 光滑均值 (C=40 窗, Θ=1) 对照实测 ==")
    meas = {600.: 0.0047, 1200.: 0.0039, 2400.: 0.0031, 4800.: 0.0023}
    sv = Sieves(60000)
    for T in [600., 1200., 2400., 4800., 9600., 19200.]:
        X = int(T / TWO_PI)
        l = math.log(T / TWO_PI)
        ell1 = l + 2 * LOG2 - 1
        norm4 = l * X * ell1 ** 4
        pp, lw = primepowers(int(math.sqrt(X)) + 1)
        th = np.exp(np.linspace(math.log(TWO_PI), math.log(C_TAPER), 1200))
        tot = 0.0
        for i in range(len(pp)):
            for k in range(i, len(pp)):
                b1, b2 = int(pp[i]), int(pp[k])
                sel = th <= T / max(b1, b2) ** 2
                if not sel.any():
                    continue
                ths = th[sel]
                q = np.array([qsym_exact(l, b1, (T / t) / b1, b2,
                                         (T / t) / b2) for t in ths])
                if b1 != b2:
                    pass  # qsym_exact 已含双序 (b≠m 腿序 + b₁b₂ 互换由
                    # 无序对枚举 i≤k 与下方权重处理)
                sarr = sv.s_array(b1, b2)
                jmax = min(int(C_TAPER / ths.min()) + 1, sv.J)
                js = np.arange(1, jmax + 1)
                ph = np.outer(ths, js)
                rows = (np.sin(2 * ph) - np.sin(ph)) / js
                rows[ph > C_TAPER] = 0.0
                integ = np.trapezoid(rows * (q / ths ** 2)[:, None],
                                     ths, axis=0)
                w = lw[i] * lw[k] / (b1 * b2)
                if b1 != b2:
                    # 无序对: (b₁,b₂) 与 (b₂,b₁) 两胞元, |·| 相同幅度
                    # (𝔖 对称, Q 双序在 qsym_exact 内已各自求和) — 但
                    # Q(b₁,b₂) ≠ Q(b₂,b₁): 各算各
                    q2 = np.array([qsym_exact(l, b2, (T / t) / b2, b1,
                                              (T / t) / b1) for t in ths])
                    integ2 = np.trapezoid(rows * (q2 / ths ** 2)[:, None],
                                          ths, axis=0)
                    tot += w * T * 2 * float(
                        np.dot(sarr[1:jmax + 1],
                               0.5 * (np.abs(integ) + np.abs(integ2))))
                    tot += w * T * 2 * float(
                        np.dot(sarr[1:jmax + 1],
                               0.5 * (np.abs(integ) + np.abs(integ2))))
                else:
                    tot += w * T * 2 * float(np.dot(sarr[1:jmax + 1],
                                                    np.abs(integ)))
        val = tot / math.pi / norm4
        m = meas.get(T)
        extra = ''
        if m:
            extra = (f"  实测={m:.4f}  残差={m - val:+.4f}  "
                     f"比={val / m:.2f}")
        print(f"T={T:.0f}: B_cellmean(C=40) = {val:.4f}{extra}")

    print("\n== taper 一致 LP 重跑 (C=∞ 实测 B_cell(Θ), l=6.64) ==")
    rho = {0.97: 0.0013, 0.98: 0.0127, 0.99: 0.0252, 1.00: 0.0387}
    cred = {0.97: 0.0150, 0.98: 0.0153, 0.99: 0.0156, 1.00: 0.0159}
    thg = [1.0, 1.5, 2.0, 3.0]
    bcell = {1.0: 0.0163, 1.5: 0.0224, 2.0: 0.0280, 3.0: 0.0364}
    print("required MID(λ,Θ) = ρ+credit−B_cell:")
    print("        " + "  ".join(f"Θ={t:3.1f}" for t in thg))
    for lam_ in [0.97, 0.98, 0.99, 1.00]:
        row = [rho[lam_] + cred[lam_] - bcell[t] for t in thg]
        print(f"λ={lam_:.2f} " + "  ".join(f"{v:+.4f}" for v in row))
    print("(存档值 λ=1,Θ=1: +0.0315; 中段带模型质量 0.004)")


def main():
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'band'
    if mode == 'band':
        T = float(sys.argv[2])
        led = band_ledger(T)
        Rdir = band_direct(T)
        LCS = math.sqrt(led['W2']) * math.sqrt(led['D'])
        Lfl = math.sqrt(led['W2']) * math.sqrt(led['diag'])
        Lth = math.sqrt(led['W2']) * math.sqrt(led['Dthm'])
        dev = led['wA'] - led['wMT']
        d0 = abs(led['wA'] - Rdir)
        print(f"T={T:.0f} 带账本: 胞元={led['ncell']}")
        print(f"  F-A0: Σw̃Ã={led['wA']:+.5f}  R_band^直接={Rdir:+.5f}  "
              f"Σ|w̃Ã|={led['abswA']:.5f}  差={d0:.5f} "
              f"({d0 / max(led['abswA'], 1e-12):.1%}) "
              f"{'✓' if d0 <= 0.2 * led['abswA'] else '✗ 鸣响'}")
        print(f"  Σw̃MT̃={led['wMT']:+.5f}  Σw̃(Ã−MT̃)={dev:+.5f}")
        print(f"  F-A1: L_CS = {LCS:.4f} vs 0.0383 → "
              f"{'GO ✓' if LCS <= 0.0383 else ('部分鸣响 (≤2×)' if LCS <= 0.0766 else '结构鸣响 (>2×)')}")
        print(f"  F-A2: L_floor(diag̃) = {Lfl:.4f}")
        print(f"  F-A3: L_thm(C_sel=4) = {Lth:.4f}")
        LCSc = math.sqrt(led['W2c']) * math.sqrt(led['Dc'])
        print(f"  [粒度阶梯] 粗胞元 (j-求和执行, {led['ncoarse']} 个): "
              f"L_CS^coarse = {LCSc:.4f} "
              f"{'GO ✓' if LCSc <= 0.0383 else ('部分鸣响' if LCSc <= 0.0766 else '结构鸣响')}")
        print(f"           Σ|w̃𝒜|^coarse = {led['abswAc']:.5f}  "
              f"(胞元级 Σ|w̃Ã| = {led['abswA']:.5f})")
        print(f"  CS-松弛: 胞元级 {LCS / max(abs(dev), 1e-12):.1f}× | "
              f"粗胞元 {LCSc / max(abs(dev), 1e-12):.1f}×")
    elif mode == 'g2':
        g2_measure()
    elif mode == 'cellmean':
        cellmean()
    print(f"总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
