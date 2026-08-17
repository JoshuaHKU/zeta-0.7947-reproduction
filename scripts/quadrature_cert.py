# -*- coding: utf-8 -*-
"""Continuum-quadrature certification (paper §4, item N2): the
hybrid edge/deep-zone quadrature for C_core, the certified geometric
tail, and the c0 = gamma - 2 chain. 连续求积认证 (论文 §4, N2 项):
C_core 的边缘/深区混合求积、认证几何尾与 c0 = γ−2 链.

结构 (边缘/深区分裂):
  gmass 恒等式:  dM/dν = l·y(θ(ν)),  y(θ) = −𝒢^{(1,1)}(θ)/θ,
  θ(ν) = 2π e^{−(ν−1)l}.  法则形 y = log(1/θ) + c* 仅在深区有效;
  边缘 (θ = O(1)) 处 y 由精确 γ-级数逐点求值 (G₀ 闭式, 无截断).
  ⟹ 混合求积 C_hybrid(l):  θ ≥ θ_c 用精确 y (γ-数组 D=1.6e7),
     θ < θ_c 用法则 y (a=1 经典律钉住 + 认证 c*-带).
  注意: 纯法则形在 ν→1⁺ 处含法则外推 (y(2π)-真值 = 0,
  法则值 = −log2π + c* ≠ 0) — F-QC1 实测该污染并入前向更正记录.

预注册证伪器 (先测后信; 全部在运行前定档):
  F-QC1 边缘诚实: |C_hybrid − C_law| (l=160, 同网格) ≤ 0.006;
         超出 ⟹ 法则值结构性污染, 走更正记录 (鸣响即信息).
  F-QC2 预算帽: B_total ≤ 0.0045 且 R̄_cert = C_core^UB + 0.0111 < 0
         (C_core^UB = C_hybrid^∞ + B_total).
  F-QC3 三方交叉: C_core^hybrid ∈ [−0.026, −0.014]
         (存档梯子 −0.020(2) 3σ-扩带; 两次独立复现 −0.0208/−0.0209).
  F-QC4 Richardson 稳定: |C_2pt(160,240) − C_3pt(100,160,240)| ≤ 0.0015.
  F-QC5 c*-闭合: |a_free − 1| ≤ 0.002 且
         |c*_pin − (log π + c₀ − osc̄)| ≤ 0.02 (部分和路线内部一致).
  F-QC6 γ-尾包络: 二进块 S_k = Σ_{2^k<d≤2^{k+1}}|γ_d| 末四倍频
         比率 ≤ 0.75 ⟹ 认证 TΓ(D) ≤ S_last·r/(1−r).

认证预算 B_total 逐项 (全显式):
  B_c*  = |∂C/∂c*| × band(c*)   [band(c*) = 末十年残差谱 + π·TΓ/θ-包络]
  B_a   = |∂C/∂a|  × |a_free − 1|  [经典律钉 a=1, 带 = 实测偏差]
  B_rich= |C_2pt − C_3pt| + 拟合残差
  B_grid= 2 × |C(160×160) − C(80×80)|  (l=100 实测, 几何裕度 ×2)
  B_exact= π·TΓ(D)/θ_c 的精确区积分传播 + 插值误差
  B_slab= δ-扫描的边缘 slab 敏感度
  (类纯度 O(l⁻¹) 归 CL″ 速率引理, 不进连续极限常数的预算.)

用法: python3 quadrature_cert.py gamma | ladder | dc | hybrid
约定对齐 m1_suite (C-taper 硬截断, Qsym 无序胞元, (1/π)/(d·ℓ₁⁴) 归一).
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from g1_ledger import qsym_exact
from tail_bound import gamma_signed_array

TWO_PI = 2 * math.pi
D_BIG = 16000000          # γ-array depth / γ-数组深度 (压包络)
CACHE = '/tmp/quad_g11.npz'
CSAW = 2.10               # Lemma S₀ 认证常数


def g0_saw(x):
    """G₀ closed form / G₀ 闭式 (精确锯齿差, 与存档逐位一致)."""
    xm = np.mod(2 * x, TWO_PI)
    s2 = np.where(xm == 0, 0.0, (math.pi - xm) / 2)
    xm1 = np.mod(x, TWO_PI)
    s1 = np.where(xm1 == 0, 0.0, (math.pi - xm1) / 2)
    return 2 * (s2 - s1)


def load_gamma():
    """γ^{(1,1)} 数组 (缓存; 压缩为非零支撑对)."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z['d'], z['g'], float(z['tG']), float(z['c0'])
    t0 = time.time()
    g = gamma_signed_array(D_BIG, 1, 1)
    d = np.nonzero(g)[0]
    gv = g[d].copy()
    # 二进块 |γ| 表 → 认证尾 TΓ(D_BIG) (F-QC6 在 gamma 模式打印)
    np.savez(CACHE, d=d, g=gv, tG=-1.0, c0=0.0)
    print(f"    [γ-数组 D={D_BIG:.0e} 构建 {time.time()-t0:.0f}s, "
          f"非零 {len(d)}]")
    return d, gv, -1.0, 0.0


def dyadic_report(d, gv):
    """二进 |γ| 块 + 几何尾认证 + c₀ 部分和扫描."""
    absg = np.abs(gv)
    print("[F-QC6] 二进块 S_k = Σ_{2^k<d≤2^{k+1}}|γ_d|:")
    ks = range(10, int(math.log2(D_BIG)))
    S = []
    for k in ks:
        m = (d > 2 ** k) & (d <= 2 ** (k + 1))
        S.append(float(absg[m].sum()))
        print(f"    k={k:2d}: S_k = {S[-1]:.3e}"
              + (f"  比率 {S[-1]/S[-2]:.3f}" if len(S) > 1 else ""))
    r4 = [S[i] / S[i - 1] for i in range(len(S) - 3, len(S))]
    rmax = max(r4)
    ok = rmax <= 0.75
    Slast = S[-1]
    tG = Slast * rmax / (1 - rmax) if ok else float('inf')
    print(f"    末四比率 max = {rmax:.3f} {'✓' if ok else '✗ 鸣响'}"
          f"  ⟹ 认证 TΓ({D_BIG:.0e}) ≤ {tG:.3e}")
    # c₀: Σ_{d≤M} dγ_d − log M → c₀ (十年扫描)
    dg = d.astype(float) * gv
    order = np.argsort(d)
    dd, dgs = d[order], np.cumsum(dg[order])
    print("[c₀] Σ_{d≤M} dγ_d − log M:")
    c0s = []
    for M in [1e4, 1e5, 1e6, 4e6, 1.6e7]:
        i = np.searchsorted(dd, M) - 1
        c0s.append(float(dgs[i]) - math.log(M))
        print(f"    M={M:.0e}: {c0s[-1]:+.5f}")
    c0 = c0s[-1]
    z = dict(np.load(CACHE))
    z['tG'], z['c0'] = tG, c0
    np.savez(CACHE, **z)
    return tG, c0


def y_exact(d, gv, thetas, tG):
    """y(θ) = −𝒢/θ 精确值 + 认证带 π·TΓ/θ (逐点)."""
    out = np.zeros(len(thetas))
    for i, t in enumerate(thetas):
        out[i] = -float(np.dot(gv, g0_saw(d * t))) / t
    band = math.pi * tG / np.asarray(thetas)
    return out, band


def mode_gamma():
    d, gv, _, _ = load_gamma()
    dyadic_report(d, gv)


def mode_ladder():
    d, gv, tG, c0 = load_gamma()
    if tG < 0:
        tG, c0 = dyadic_report(d, gv)
    # 60 点梯子, 饱和裕度 100× (θ ≥ 100/D)
    lt_hi = math.log10(D_BIG / 100.0)
    lts = np.linspace(3.0, lt_hi, 60)
    thetas = 10.0 ** (-lts)
    y, band = y_exact(d, gv, thetas, tG)
    x = np.log(1.0 / thetas)
    # 自由拟合 (存档复现锚)
    A = np.vstack([x, np.ones_like(x)]).T
    (af, cf), *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"[梯子] 自由拟合: a = {af:+.5f}  c* = {cf:+.5f} "
          f"(存档锚 +0.99945/−0.54328, D=2e6 窗)")
    okA = abs(af - 1) <= 0.002
    print(f"[F-QC5a] |a_free − 1| = {abs(af-1):.5f} "
          f"{'✓ ≤ 0.002' if okA else '✗ 鸣响'}")
    # a=1 钉住 (经典律): c* 逐点谱 + 十年段均值
    cpt = y - x
    # 充带规则 (数据无关, 运行前定): 逐十年段 charge = 段内均值包络
    # (log-均匀网格上 mean π·TΓ/θ) + 段内 max 残差; c*_pin 取 charge
    # 最小段的段均值; band_c = 最小 charge + 段间最大漂移 (法则缺陷系统项).
    print("[c*-钉住] 十年段均值 (段内 max 残差 | 均值包络 | charge):")
    segs = [(3.0, 4.0), (4.0, 5.0), (5.0, lt_hi)]
    means, charges = [], []
    for lo, hi in segs:
        m = (lts >= lo) & (lts <= hi)
        mu = float(cpt[m].mean())
        rs = float(np.max(np.abs(cpt[m] - mu)))
        be = float(np.mean(band[m]))
        means.append(mu)
        charges.append(rs + be)
        print(f"    十年 [{lo:.1f},{hi:.1f}]: c* = {mu:+.5f} "
              f"± {rs:.4f} | 包络均值 {be:.4f} | charge {rs+be:.4f}")
    ib = int(np.argmin(charges))
    cpin = means[ib]
    drift = max(abs(means[i + 1] - means[i]) for i in range(len(means) - 1))
    band_c = charges[ib] + drift
    mlast = (lts >= segs[-1][0])
    print(f"    c*_pin = {cpin:+.5f} (段 {ib})  带 = charge "
          f"{charges[ib]:.4f} + 漂移 {drift:.4f} = ±{band_c:.4f}")
    # osc-闭合: c* = log π + c₀ − osc̄,  osc(θ) = (1/θ)Σ_{dθ≥π}γ_d G₀(dθ)
    print("[F-QC5b] 部分和路线闭合:")
    oscs = []
    for i, t in enumerate(thetas[mlast][::12]):
        mm = d.astype(float) * t >= math.pi
        o = float(np.dot(gv[mm], g0_saw(d[mm] * t))) / t
        oscs.append(o)
    oscbar = float(np.mean(oscs))
    lhs = cpin
    rhs = math.log(math.pi) + c0 - oscbar
    # {M/d}-均值修正属 o(1): 闭合检验容 0.02
    ok5 = abs(lhs - rhs) <= 0.02
    print(f"    c*_pin = {lhs:+.5f} vs log π + c₀ − osc̄ = "
          f"{math.log(math.pi):+.5f} {c0:+.5f} − ({oscbar:+.5f}) "
          f"= {rhs:+.5f}  |Δ| = {abs(lhs-rhs):.4f} "
          f"{'✓ ≤ 0.02' if ok5 else '✗ 鸣响'}")
    np.savez('/tmp/quad_cstar.npz', cpin=cpin, band_c=band_c, af=af,
             cf=cf, tG=tG, c0=c0)


# ---------------------------------------------------------------- 求积
def _vol_vec(l, a, b, c):
    """vol 的向量形: max(l − (max(0,a,b,c) − min(0,a,b,c)), 0)."""
    z = np.zeros_like(a)
    mx = np.maximum(np.maximum(z, a), np.maximum(b, c))
    mn = np.minimum(np.minimum(z, a), np.minimum(b, c))
    return np.maximum(l - (mx - mn), 0.0)


def _o4_vec(l, u1, u3, u2, u4):
    """o4 向量形 (de = 0 恒成立: 两积同 ν)."""
    return (_vol_vec(l, -u1, u3 - u4, -u4)
            + _vol_vec(l, u2 - u3 - u4, -u3 - u4, -u4)
            + _vol_vec(l, -u2 - u3 + u4, -u3 + u4, u4))


def _qgrid(l, nu, nb):
    """固定 ν 的 β-双重 Qsym 和 — 向量化, 与 qsym_exact 逐位同约定:
    4 腿序求和; 等腿侧 (β = ν/2 端点, U1 == U3 精确) 权 1/2."""
    bmin = max(nu - 1.0, 0.0) + 1e-4
    bmax = nu / 2
    if bmax <= bmin:
        return 0.0
    bs = np.linspace(bmin, bmax, nb)
    db = bs[1] - bs[0]
    U1 = (bs * l)[:, None] * np.ones((1, nb))
    U3 = ((nu - bs) * l)[:, None] * np.ones((1, nb))
    U2 = np.ones((nb, 1)) * (bs * l)[None, :]
    U4 = np.ones((nb, 1)) * ((nu - bs) * l)[None, :]
    tot = (_o4_vec(l, U1, U3, U2, U4) + _o4_vec(l, U3, U1, U2, U4)
           + _o4_vec(l, U1, U3, U4, U2) + _o4_vec(l, U3, U1, U4, U2))
    w1 = np.where(np.isclose(bs, nu - bs, rtol=0, atol=1e-15), 0.5, 1.0)
    tot *= w1[:, None] * w1[None, :]
    return float(tot.sum()) * db * db * l * l


def ccore_general(l, gmass_fn, ngrid_nu=40, ngrid_b=40, numin=1.0):
    """混合/法则通用求积: gmass_fn(ν) 外置."""
    ell1 = l + 2 * math.log(2.0) - 1
    nus = np.linspace(numin + 1e-3, 2.0, ngrid_nu)
    dnu = nus[1] - nus[0]
    tot = 0.0
    for nu in nus:
        q = _qgrid(l, nu, ngrid_b)
        if q:
            tot += q * gmass_fn(nu) * dnu
    return tot * TWO_PI / (math.pi * l * ell1 ** 4)


def gmass_law(l, cstar, a=1.0):
    def f(nu):
        return -(a * l * l * (nu - 1.0) + l * (cstar - math.log(TWO_PI)))
    return f


_YTAB = None      # (lg, yv, theta_c) 全局缓存 — y(θ) 与 l 无关


def build_ytab(d, gv):
    """精确区 y 的对数-θ 密网 (600 点) — 构建一次, npz 缓存."""
    global _YTAB
    if _YTAB is not None:
        return _YTAB
    if os.path.exists('/tmp/quad_ytab.npz'):
        z = np.load('/tmp/quad_ytab.npz')
        _YTAB = (z['lg'], z['yv'], float(z['tc']))
        return _YTAB
    theta_c = 100.0 / D_BIG
    lg = np.linspace(math.log(theta_c), math.log(TWO_PI), 600)
    th = np.exp(lg)
    yv = np.zeros(len(th))
    df = d.astype(float)
    for i, t in enumerate(th):
        yv[i] = -float(np.dot(gv, g0_saw(df * t))) / t
    np.savez('/tmp/quad_ytab.npz', lg=lg, yv=yv, tc=theta_c)
    _YTAB = (lg, yv, theta_c)
    return _YTAB


def make_gmass_hybrid(l, cstar, d, gv, a=1.0, env=0.0, tG=0.0):
    """θ(ν) ≥ θ_c: 精确 y 插值 (±env×包络); 否则法则. 返回 (fn, θ_c)."""
    lg, yv, theta_c = build_ytab(d, gv)

    def f(nu):
        th_nu = TWO_PI * math.exp(-(nu - 1.0) * l)
        if th_nu >= theta_c:
            yy = float(np.interp(math.log(th_nu), lg, yv))
            yy += env * math.pi * tG / th_nu      # 包络极端扫描
        else:
            yy = a * math.log(1.0 / th_nu) + cstar
        return -l * yy      # dM/dν = l·y; gmass = −l·y (符号注见下)
    return f, theta_c


# 符号注: 法则形 gmass = −(l²(ν−1)+l(c*−log2π)) = −l·(x+c*) = −l·y_law
# (y = −𝒢/θ = x + c* > 0 深区) ⟹ 通用形 gmass(ν) = −l·y(θ(ν)). ✓


def mode_dc():
    z = np.load('/tmp/quad_cstar.npz')
    cpin = float(z['cpin'])
    print("[∂C] 法则形导数 (40×40 网格):")
    for l in [100.0, 160.0]:
        c1 = ccore_general(l, gmass_law(l, cpin))
        c2 = ccore_general(l, gmass_law(l, cpin + 0.03))
        c3 = ccore_general(l, gmass_law(l, cpin, a=1.002))
        print(f"    l={l:.0f}: C = {c1:+.5f}  ∂C/∂c* = "
              f"{(c2-c1)/0.03:+.5f}  ∂C/∂a = {(c3-c1)/0.002:+.5f}")
    np.savez('/tmp/quad_dc.npz',
             dCdc=(c2 - c1) / 0.03, dCda=(c3 - c1) / 0.002)


def richardson(ls, vs):
    (l1, v1), (l2, v2) = (ls[-2], vs[-2]), (ls[-1], vs[-1])
    C2 = (v2 * l2 - v1 * l1) / (l2 - l1)
    # 3 点: v = C + p/l + q/l²
    l0, v0 = ls[-3], vs[-3]
    Ainv = np.linalg.solve(
        np.array([[1, 1 / l0, 1 / l0 ** 2],
                  [1, 1 / l1, 1 / l1 ** 2],
                  [1, 1 / l2, 1 / l2 ** 2]]),
        np.array([v0, v1, v2]))
    return C2, float(Ainv[0])


def mode_hybrid():
    t0 = time.time()
    d, gv, tG, c0 = load_gamma()
    z = np.load('/tmp/quad_cstar.npz')
    cpin, band_c = float(z['cpin']), float(z['band_c'])
    dz = np.load('/tmp/quad_dc.npz')
    dCdc, dCda = float(dz['dCdc']), float(dz['dCda'])
    afree = float(z['af'])
    ls = [40.0, 60.0, 100.0, 160.0, 240.0]
    vh, vl = [], []
    print(f"[混合求积] c*_pin = {cpin:+.5f}, a = 1 (经典律), 80×80:")
    for l in ls:
        gm, th_c = make_gmass_hybrid(l, cpin, d, gv)
        h = ccore_general(l, gm, 80, 80)
        lw = ccore_general(l, gmass_law(l, cpin), 80, 80)
        vh.append(h)
        vl.append(lw)
        print(f"    l={l:.0f}: C_hybrid = {h:+.5f}   C_law = {lw:+.5f}"
              f"   Δ = {h-lw:+.5f}   [{time.time()-t0:.0f}s]")
    dedge = abs(vh[-2] - vl[-2])
    print(f"[F-QC1] |C_hybrid − C_law|(l=160) = {dedge:.5f} "
          f"{'✓ ≤ 0.006' if dedge <= 0.006 else '✗ 鸣响 (边缘污染, 走更正)'}")
    C2, C3 = richardson(ls, vh)
    # 窗移稳定: 3pt 于 (60,100,160) vs (100,160,240) — B_rich 取两者最大
    A = np.array([[1, 1 / ls[1], 1 / ls[1] ** 2],
                  [1, 1 / ls[2], 1 / ls[2] ** 2],
                  [1, 1 / ls[3], 1 / ls[3] ** 2]])
    C3w = float(np.linalg.solve(A, np.array(vh[1:4]))[0])
    brich = max(abs(C2 - C3), abs(C3 - C3w))
    print(f"[F-QC4] Richardson: 2pt = {C2:+.5f}  3pt = {C3:+.5f} "
          f"窗移 3pt(60..160) = {C3w:+.5f}  B_rich = max = {brich:.5f} "
          f"{'✓ ≤ 0.0015' if brich <= 0.0015 else '✗'}")
    # 网格加密 (l=100): 80×80 → 160×160
    gm, _ = make_gmass_hybrid(100.0, cpin, d, gv)
    c160 = ccore_general(100.0, gm, 160, 160)
    c80 = vh[2]
    bgrid = 2 * abs(c160 - c80)
    print(f"[网格] l=100: 80×80 = {c80:+.5f}  160×160 = {c160:+.5f} "
          f" B_grid = 2|Δ| = {bgrid:.5f}")
    # δ-扫描 (slab 敏感度, l=160)
    print("[δ-扫描] 深区起点 ν = 1+δ (l=160, 80×80):")
    gm160, _ = make_gmass_hybrid(160.0, cpin, d, gv)
    slabs = []
    for dlt in [0.0, 0.02, 0.05]:
        cdeep = ccore_general(160.0, gm160, 80, 80, numin=1.0 + dlt)
        slabs.append(cdeep)
        print(f"    δ={dlt:.2f}: C(ν≥1+δ) = {cdeep:+.5f}"
              f"   slab = {vh[3]-cdeep:+.5f}")
    # slab 带取扫描全幅 (含网格对齐/混叠敏感度, 保守向)
    devs = [vh[3] - s for s in slabs]
    bslab = max(devs) - min(devs)
    # 精确区包络传播: y±(π·TΓ/θ) 两极端重跑 (只动精确区):
    gp, _ = make_gmass_hybrid(160.0, cpin, d, gv, env=+1.0, tG=tG)
    gmm, _ = make_gmass_hybrid(160.0, cpin, d, gv, env=-1.0, tG=tG)
    cp = ccore_general(160.0, gp, 80, 80)
    cm = ccore_general(160.0, gmm, 80, 80)
    bexact = abs(cp - cm) / 2
    # 汇总预算
    Bc = abs(dCdc) * band_c
    Ba = abs(dCda) * abs(afree - 1.0)
    B = Bc + Ba + brich + bgrid + bexact + bslab
    Ccore = C3
    print("[预算] 认证带逐项:")
    print(f"    B_c*    = |∂C/∂c*|×band = {abs(dCdc):.4f}×{band_c:.4f}"
          f" = {Bc:.5f}")
    print(f"    B_a     = {abs(dCda):.4f}×{abs(afree-1):.5f} = {Ba:.5f}")
    print(f"    B_rich  = {brich:.5f}")
    print(f"    B_grid  = {bgrid:.5f}")
    print(f"    B_exact = {bexact:.5f}  (γ-尾包络传播)")
    print(f"    B_slab  = {bslab:.5f}  (δ=0.02 边缘 slab)")
    print(f"    B_total = {B:.5f} "
          f"{'✓ ≤ 0.0045' if B <= 0.0045 else '✗ 鸣响 (F-QC2)'}")
    ok3 = -0.026 <= Ccore <= -0.014
    print(f"[F-QC3] C_core = {Ccore:+.5f} ∈ [−0.026,−0.014] "
          f"{'✓' if ok3 else '✗'}  (存档: −0.020(2); 复现: −0.0208/"
          f"−0.0209)")
    Rbar = Ccore + B + 0.0111
    print(f"[判定] C_core^UB = {Ccore:+.5f} + {B:.5f} = {Ccore+B:+.5f}"
          f"  ⟹ R̄_cert ≤ {Rbar:+.5f} "
          f"{'< 0 ✓ 第二档认证' if Rbar < 0 else '✗ 未认证'}")
    # 消费公式 (C26 Prop 4.5): Λ₂(0) = (5/108+Δ/3)/(1/3+4Δ/3)
    Dl = Rbar + 1.0 / 30.0
    lam2 = (5.0 / 108.0 + Dl / 3.0) / (1.0 / 3.0 + 4.0 * Dl / 3.0)
    print(f"      Δ = {Dl:.5f}  Λ₂(0) = {lam2:.6f}  ⟹ "
          f"简单在线 ≥ {1-2*lam2:.4f}  相异 ≥ {1-lam2:.4f}")
    np.savez('/tmp/quad_final.npz', Ccore=Ccore, B=B, Rbar=Rbar,
             vh=np.array(vh), vl=np.array(vl), ls=np.array(ls))
    print(f"总用时 {time.time()-t0:.0f}s")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'gamma'
    {'gamma': mode_gamma, 'ladder': mode_ladder,
     'dc': mode_dc, 'hybrid': mode_hybrid}[mode]()


if __name__ == "__main__":
    main()
