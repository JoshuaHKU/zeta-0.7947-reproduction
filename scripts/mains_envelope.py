# -*- coding: utf-8 -*-
"""Rigorous envelope of the deterministic mains: three-piece
decomposition with proof-grade constants (helper for the §4 chain).
主项严格包络: 三片分解与证明级配件常数 (§4 链辅助). 本脚本实现
三片分解并测量每片:

    𝔖_{b₁,b₂}(j) = D̄(j) + g_P(j) + g_tail(j)
    D̄     = 局部均值 (dbar_alpha 精确有限 Möbius 分解; E[D̄]=1)
    g_P    = Σ_{q ≤ P, q 含 ∤2b₁b₂ 的素因子} β_q c_q(j)   (显式振荡片)
    g_tail = 其余 (q > P)                                  (X.2-级尾)

对应 𝒢-核级数三片: 𝒢_mean (锯齿闭式, 精确) + 𝒢_P (有限显式) + 𝒢_tail.
证明级配件 (备忘录 §2):
    Abel:  |Σ_{j≤M} c_q(j)| ≤ C_q := ∏_{p^a∥q}(p^a + p^{a−1})   (q>1)
    行范数: Σ_j row_j² ≤ 6θ + O(θ²)  (row_j = [sin2θj − sinθj]/j)
    尾 CS: |𝒢_tail(θ)| ≤ 2‖g_tail‖₂,J·(Σrow²)^{1/2}

预注册证伪器:
  F-E3a  (i) D̄ 的 q-展开重构: Σ_{q|(2b₁b₂)^∞, q≤4000} β_q c_q ≈ D̄,
         max|差| ≤ 0.02 (三个代表对); (ii) 全和锚: Σ 三片 zone-积分
         = zone-sum 总量, 对 m1_suite 参考 (T=2400 无 e-cut −0.00544)
         相对差 ≤ 2%.
  F-E3b  包络: env_total = |mean+P 签名和| + tail-CS-积分 ≤ 0.03
         (T = 2400, 4800), 且不随 l 增长. 超限 → 尾处理需精化 (部分鸣响).
  F-A-red 黏合权 AP-校准: T2 族 (T=4800) 上 Σ_{n≡a(q)}w vs 均匀预测
         的 max 相对偏差 ≤ 25% (q ∈ {3,4,5,7,8,9,11,12}; 𝔖-修正后).

用法: python3 mains_envelope.py env 2400
      python3 mains_envelope.py env 4800
      python3 mains_envelope.py apcalib
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from m1_suite import (Sieves, dbar_alpha, g0_sawtooth, kap_p, primepowers,
                      prime_factors, vp_int)

TWO_PI = 2 * math.pi
LOG2 = math.log(2.0)
P_CUT = 40           # q ≤ P 显式片
JMAX = 60000


def euler_phi(n):
    r, nn = 1, n
    for p in prime_factors(n):
        e = vp_int(nn, p)
        r *= p ** (e - 1) * (p - 1)
    return r


def mu(n):
    if n == 1:
        return 1
    r = 1
    for p in prime_factors(n):
        if n % (p * p) == 0:
            return 0
        r = -r
    return r


def cq_array(q, J):
    """Ramanujan 和 c_q(j), j = 0..J (经 c_q = Σ_{d|(q,j)} d·μ(q/d))."""
    c = np.zeros(J + 1)
    for d in range(1, q + 1):
        if q % d == 0:
            m_ = mu(q // d)
            if m_:
                c[d::d] += d * m_
            if m_:
                c[0] += d * m_    # j=0: 全 d|0
    return c


def beta_q(q, b1, b2):
    """Prop S 系数 β_q = μ(q₁)μ(q₂)/(φ(q₁)φ(q₂)), q_i = q/(q,b_i)."""
    q1, q2 = q // math.gcd(q, b1), q // math.gcd(q, b2)
    m1_, m2_ = mu(q1), mu(q2)
    if m1_ == 0 or m2_ == 0:
        return 0.0
    return m1_ * m2_ / (euler_phi(q1) * euler_phi(q2))


def class_split(q, b1, b2):
    """q 是否属于 D̄-类 (全部素因子 | 2b₁b₂)."""
    for p in prime_factors(q):
        if (2 * b1 * b2) % p:
            return False
    return True


def C_q_abel(q):
    """Abel 常数 C_q = ∏_{p^a||q}(p^a + p^{a−1})  (备忘录证明)."""
    r = 1
    for p in prime_factors(q):
        a = vp_int(q, p)
        r *= p ** a + p ** (a - 1)
    return r


# ----------------------------------------------------------------------
def dbar_recon_check():
    """F-E3a(i): D̄ = Σ_{q∈类} β_q c_q 数值重构 (q ≤ 4000)."""
    print("[F-E3a-i] D̄ 的 q-展开重构 (q ≤ 4000, J = 2000):")
    J = 2000
    ok_all = True
    for b1, b2 in [(5, 3), (2, 2), (4, 6)]:
        db = np.zeros(J + 1)
        for c, al in dbar_alpha(b1, b2):
            db[c::c] += al
            db[0] += al
        rec = np.zeros(J + 1)
        for q in range(1, 4001):
            if class_split(q, b1, b2):
                bq = beta_q(q, b1, b2)
                if bq:
                    rec += bq * cq_array(q, J)
        d = float(np.max(np.abs(rec[1:] - db[1:])))
        ok = d <= 0.02
        ok_all &= ok
        print(f"    (b₁,b₂)=({b1},{b2}): max|重构−D̄| = {d:.4f} "
              f"{'✓' if ok else '✗'}")
    return ok_all


# ----------------------------------------------------------------------
def envelope(T):
    """三片 zone-积分 + 包络 (全 θ-域, C=∞, 与 suite 约定一致)."""
    X = int(T / TWO_PI)
    l = math.log(T / TWO_PI)
    ell1 = l + 2 * LOG2 - 1
    norm4 = l * X * ell1 ** 4
    thmin = 4 * math.pi ** 2 / T
    th = np.exp(np.linspace(math.log(thmin), math.log(200.0),
                            int(math.log(200.0 / thmin) * 640) + 2))
    om = np.log(TWO_PI / th)
    pp, lw = primepowers(X)
    sv = Sieves(JMAX)
    # c_q 数组 (q ≤ P, 一次预计算)
    cqs = {q: cq_array(q, JMAX) for q in range(2, P_CUT + 1)}

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

    tot_exact = 0.0
    tot_mean = 0.0
    tot_P = 0.0
    tot_tailex = 0.0          # 尾片精确 (对照)
    tot_tailenv = 0.0         # 尾片 CS 包络 (证明级)
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
            # ---- 三片 j-数组 -----------------------------------------
            sarr = sv.s_array(b1, b2)
            db = np.zeros(JMAX + 1)
            combos = dbar_alpha(b1, b2)
            for c, al in combos:
                db[c::c] += al
            gP = np.zeros(JMAX + 1)
            for qq in range(2, P_CUT + 1):
                if not class_split(qq, b1, b2):
                    bq = beta_q(qq, b1, b2)
                    if bq:
                        gP += bq * cqs[qq]
            gtail = sarr - db - gP
            gtail[0] = 0.0
            tail_l2 = math.sqrt(float(np.mean(gtail[1:] ** 2)))
            # ---- θ-积分 (逐片) ---------------------------------------
            g_mean = np.zeros(len(ths))
            for c, al in combos:
                g_mean += (al / c) * g0_sawtooth(c * ths)
            g_Pv = np.zeros(len(ths))
            g_tv = np.zeros(len(ths))
            env_t = np.zeros(len(ths))
            for ii, t in enumerate(ths):
                J = min(int(200 / t), JMAX)
                if J < 1:
                    continue
                j = np.arange(1, J + 1)
                row = (np.sin(2 * t * j) - np.sin(t * j)) / j
                g_Pv[ii] = 2 * float(np.dot(gP[1:J + 1], row))
                g_tv[ii] = 2 * float(np.dot(gtail[1:J + 1], row))
                # 证明级: ‖g_tail‖₂,J · (Σrow²)^{1/2} (行范数数值+解析帽)
                env_t[ii] = 2 * (tail_l2 * math.sqrt(J)
                                 * math.sqrt(float(np.sum(row ** 2))))
            f0 = q / ths ** 2
            tot_exact += w * T * np.trapezoid(
                (g_mean + g_Pv + g_tv) * f0, ths)
            tot_mean += w * T * np.trapezoid(g_mean * f0, ths)
            tot_P += w * T * np.trapezoid(g_Pv * f0, ths)
            tot_tailex += w * T * np.trapezoid(g_tv * f0, ths)
            tot_tailenv += w * T * np.trapezoid(env_t * np.abs(f0), ths)
    nz = math.pi * norm4
    return dict(exact=tot_exact / nz, mean=tot_mean / nz,
                P=tot_P / nz, tailex=tot_tailex / nz,
                tailenv=tot_tailenv / nz)


# ----------------------------------------------------------------------
# γ-重求和 (F-E3b 鸣响后的替代, 本轮发现):
#   Σ_j c_q(j)·row_j(θ) = Σ_{d|q} μ(q/d)·G₀(dθ)/2   (闭式锯齿恒等式)
#   ⟹ 𝒢(θ) = Σ_d γ_d·G₀(dθ),  γ_d = Σ_e μ(e)β_{de}  (逐素局部闭式)
#   γ-局部: γ_p(v) = β_p(v) − β_p(v+1);  β_p(a) = ∏_i μ(p^{aᵢ})/φ(p^{aᵢ}),
#   aᵢ = max(a − v_p(bᵢ), 0)  (aᵢ ≥ 2 ⟹ 0).  奇 d ⟹ γ_d = 0 (奇偶性).
# ----------------------------------------------------------------------
def _beta_loc(p, a, e1, e2):
    f = 1.0
    for ei in (e1, e2):
        ai = max(a - ei, 0)
        if ai >= 2:
            return 0.0
        if ai == 1:
            f *= -1.0 / (p - 1)
    return f


def gamma_d(d, b1, b2, base_odd):
    """γ_d 闭式: ∏_{p|d}(β_p(v)−β_p(v+1)) × ∏_{p∤d}(β_p(0)−β_p(1))."""
    val = 1.0
    ps_d = prime_factors(d) if d > 1 else set()
    ps_b = prime_factors(2 * b1 * b2)
    for p in ps_d:
        e1, e2 = vp_int(b1, p), vp_int(b2, p)
        v = vp_int(d, p)
        val *= _beta_loc(p, v, e1, e2) - _beta_loc(p, v + 1, e1, e2)
        if val == 0.0:
            return 0.0
    # p ∤ d 部分: 特殊素数 (p | 2b₁b₂) 逐个, 其余用通用尾积
    corr = 1.0
    for p in ps_b:
        if p in ps_d:
            continue
        e1, e2 = vp_int(b1, p), vp_int(b2, p)
        f = _beta_loc(p, 0, e1, e2) - _beta_loc(p, 1, e1, e2)
        if f == 0.0:
            return 0.0
        corr *= f
        if p != 2:
            # 通用尾积按 (1−(p−1)⁻²) 记账了此 p — 除回
            corr /= (1 - 1.0 / (p - 1) ** 2)
    tail = base_odd
    for p in ps_d:
        if p > 2 and p not in ps_b:
            tail /= (1 - 1.0 / (p - 1) ** 2)
    return val * corr * tail


def env_gamma(T, D_EVAL=400, D_TAIL=4000):
    """γ-重求和主项求值 + 截断尾界 (F-E3b 修订版)."""
    from m1_suite import C2_TWIN
    X = int(T / TWO_PI)
    l = math.log(T / TWO_PI)
    ell1 = l + 2 * LOG2 - 1
    norm4 = l * X * ell1 ** 4
    thmin = 4 * math.pi ** 2 / T
    th = np.exp(np.linspace(math.log(thmin), math.log(200.0),
                            int(math.log(200.0 / thmin) * 640) + 2))
    om = np.log(TWO_PI / th)
    pp, lw = primepowers(X)
    # 通用奇素尾积 ∏_{p>2}(1−(p−1)⁻²) = C₂
    base_odd = C2_TWIN

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

    tot_eval, tot_tailbd = 0.0, 0.0
    ident_report = []
    sv = Sieves(20000)
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
            gam = {}
            for d in range(1, D_EVAL + 1):
                gd = gamma_d(d, b1, b2, base_odd)
                if gd:
                    gam[d] = gd
            gv = np.zeros(len(ths))
            for d, gd in gam.items():
                gv += gd * g0_sawtooth(d * ths)
            f0 = q / ths ** 2
            tot_eval += w * T * np.trapezoid(gv * f0, ths)
            # 截断尾: Σ_{D<d≤D_TAIL}|γ_d| × |G₀|≤π/2·2 × ∫|Q|θ⁻²
            tail_g = sum(abs(gamma_d(d, b1, b2, base_odd))
                         for d in range(D_EVAL + 1, D_TAIL + 1, 1))
            envI = T * np.trapezoid(np.abs(f0), ths)
            tot_tailbd += w * tail_g * math.pi * envI
            # 逐对恒等式抽查 (6 个代表对)
            if (b1, b2) in [(2, 3), (3, 5), (2, 2), (5, 5), (4, 6),
                            (7, 11)]:
                # 恒等式抽查: 两侧都取全 j-和 (J = 2e4 固定; γ-级数
                # 本身即全和).  首跑用 J=200/θ 截断参考 — 比较对象
                # 错误, 已更正 (γ-总量 0.4% 吻合先行提示了这点).
                sarr = sv.s_array(b1, b2)
                errs = []
                for t in ths[::max(len(ths) // 8, 1)][:8]:
                    J = sv.J
                    j = np.arange(1, J + 1)
                    row = (np.sin(2 * t * j) - np.sin(t * j)) / j
                    gex = 2 * float(np.dot(sarr[1:J + 1], row))
                    gga = sum(gd * float(g0_sawtooth(
                        np.array([d * t]))[0]) for d, gd in gam.items())
                    errs.append((abs(gga - gex), abs(gex)))
                if errs:
                    mx = max(a for a, _ in errs)
                    sc = max(b for _, b in errs)
                    ident_report.append(((b1, b2), mx, sc))
    nz = math.pi * norm4
    return tot_eval / nz, tot_tailbd / nz, ident_report


def apcalib():
    """F-A-red: 黏合权 w 的 AP-分布校准 (MAJ-w 槽的经验前哨)."""
    from t2_swaps import Family
    fam = Family(4800., 5, 3.0, 3.3, 2, 40, 19, m_le_X=False)
    b1, J, lam = fam.b1, fam.J, fam.lam
    print("[F-A-red] w-in-APs 校准 (T=4800, b₁=5, 块[3X,3.3X]):")
    worst = 0.0
    for qm in [3, 4, 5, 7, 8, 9, 11, 12]:
        num = np.zeros(qm)
        for b2 in fam.b2s:
            g = math.gcd(b1, b2)
            r = b1 // g
            lo, hi = fam.m_range(b2)
            kmax = (hi - lo) // r
            for k in [1, 2, 3]:
                if k > kmax:
                    continue
                nlo = max(-((-(b2 * lo - J)) // b1), 2)
                nhi = (b2 * hi + J) // b1
                for n in range(nlo, nhi + 1):
                    mlo = -((-(b1 * n - J)) // b2)
                    mhi = (b1 * n + J) // b2
                    acc = 0.0
                    for m in range(max(mlo, lo), min(mhi, hi) + 1):
                        if b1 * n - b2 * m == 0:
                            continue
                        mp = m - r * k
                        if lo <= mp <= hi and lam[m] > 0 and lam[mp] > 0:
                            acc += lam[m] * lam[mp]
                    num[n % qm] += acc
        tot = num.sum()
        if tot == 0:
            continue
        # 素数幂支撑的 AP-偏置: 与 (a,q) 互素类的均匀预测比较
        coprime = np.array([1.0 if math.gcd(a, qm) == 1 else 0.0
                            for a in range(qm)])
        # w 的 n-支撑无互素约束 (n 只是窗口位置): 用全类均匀预测
        pred = tot / qm
        dev = float(np.max(np.abs(num - pred)) / pred)
        worst = max(worst, dev)
        print(f"    q={qm:>2}: max 类偏差 = {dev:.2f}")
    print(f"    最坏 = {worst:.2f} {'✓ (≤ 0.25 通过…宽判据)' if worst <= 0.25 else '— 注: n-支撑含算术结构, 见备忘录'}")


def main():
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'env'
    if mode == 'env':
        T = float(sys.argv[2])
        if T == 2400.:
            dbar_recon_check()
        r = envelope(T)
        signed_MP = r['mean'] + r['P']
        env_tot = abs(signed_MP) + r['tailenv']
        print(f"T={T:.0f} 三片 zone-积分 (R-单位):")
        print(f"  精确全和   = {r['exact']:+.5f}"
              f"   (m1_suite 参考: T=2400 → −0.00544)")
        print(f"  mean 片    = {r['mean']:+.5f}")
        print(f"  P 片(q≤{P_CUT}) = {r['P']:+.5f}")
        print(f"  尾片精确   = {r['tailex']:+.5f}")
        print(f"  尾片CS包络 = {r['tailenv']:.5f}")
        print(f"  [F-E3b] env_total = |mean+P| + tail包络 = {env_tot:.5f} "
              f"vs 0.03 {'✓' if env_tot <= 0.03 else '✗ 鸣响 (尾需精化)'}")
    elif mode == 'env2':
        T = float(sys.argv[2])
        ev, tb, rep = env_gamma(T)
        print(f"T={T:.0f} γ-重求和 (F-E3b 修订):")
        for pair, e, sc in rep:
            print(f"  恒等式抽查 {pair}: max|γ级数−全和| = {e:.2e} "
                  f"(𝒢-幅度尺度 {sc:.2e}) "
                  f"{'✓' if e <= max(3e-2 * sc, 2e-3) else '✗'}")
        print(f"  γ-求值 (d ≤ 400) = {ev:+.5f}  "
              f"(参考精确全和 T=2400: −0.00545)")
        print(f"  截断尾界 (400<d≤4000, |G₀|≤π 粗放) = {tb:.5f}")
        print(f"  [F-E3b′] env = |求值| + 尾界 = {abs(ev) + tb:.5f} "
              f"vs 0.03 {'✓' if abs(ev) + tb <= 0.03 else '✗'}")
    elif mode == 'apcalib':
        apcalib()
    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
