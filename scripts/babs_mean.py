# -*- coding: utf-8 -*-
"""B_abs direct pipeline + PNT-mean partial evaluation (helper for
the §4 chain). B_abs 直接管线与 PNT-均值部分求值 (§4 链辅助).

低 zone 价格 B_abs(Θ=1) ≈ 0.0231 是纯三角不等式粗界; 其均值部分
(把 Λ-质量换成 PNT 平滑密度后的同一绝对值积分) 是可证可求值的.
若均值 B_mean 承载大部分质量且残差随 l 收缩, 则认证上界可换成
[B_mean 渐近值 + 残差裕度] — 第一档纪录直接受益.

约定 (与 R-类测量管线/m1_suite 一致, λ=1, C=∞ 全核):
    四元组 = 有序对 (n₁,n₃) × (n₂,n₄), 全部素数幂 ≤ X = ⌊T/2π⌋
    N₁ = n₁n₃, N₂ = n₂n₄, N₁ ≠ N₂
    c = Λ⁴/√(N₁N₂) · Ŵin(log N₁/N₂) · O₄(u₁,u₃,u₂,u₄; L=l)
    B_abs(Θ) = (1/π)/(l·X·ℓ₁⁴) · Σ_{min(N₁,N₂)≤ΘX} |c|
    signed(Θ) 同式无绝对值;  full-R = signed(Θ=∞)

均值模型: Λ-和 → ∫e^u du (PNT), j-格 → 连续 (E[𝔖]=1),
    B_mean(Θ) = (1/π)/(l·X·ℓ₁⁴)·∫∫∫∫ e^{Σu/2}|Ŵin(Δ)|O₄·1[min≤ΘX]du⁴
    (|Δ| ≥ 1/max(N₁,N₂) 模拟 j≥1 格;  核用周期平均+精确 I₀ 表)

预注册证伪器:
  F-B1 复现: B_abs(Θ=2) 四高度 vs 存档表 (.03887/.03823/.03582/.03503)
       相对差 ≤ 2%; signed 低zone vs (+.00335/+.00154/+.00095/+.00082)
       差 ≤ 5%+2e-4; full-R (T=2400) ≈ −0.0048±0.0008 (阶段总结).
       存档 Θ-网格 (Θ=1: 0.0231) 与本管线的偏离仅记录 (其高度未注明).
  F-B2 均值追踪: |B_mean−B_abs|/B_abs ≤ 0.25 每高度, 且残差
       B_abs−B_mean 在 l ≥ 5.2 后不增. 违反 ⟹ 均值路线鸣响.

用法: python3 babs_mean.py direct 600 1200 2400
      python3 babs_mean.py direct 4800
      python3 babs_mean.py direct-restricted 9600
      python3 babs_mean.py mean
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from m1_suite import primepowers

TWO_PI = 2 * math.pi
LOG2 = math.log(2.0)
THETA_GRID = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])


def leg_pairs(X, prod_cap=None):
    """有序素数幂对 (nₐ, n_b), nₐ,n_b ≤ X, 可选乘积上限. 返回数组."""
    pp, lw = primepowers(X)
    na, nb, wa, wb = [], [], [], []
    for i, a in enumerate(pp):
        for k, b in enumerate(pp):
            if prod_cap is not None and a * b > prod_cap:
                continue
            na.append(a)
            nb.append(b)
            wa.append(lw[i])
            wb.append(lw[k])
    na = np.array(na, dtype=np.float64)
    nb = np.array(nb, dtype=np.float64)
    w = np.array(wa) * np.array(wb)
    return na, nb, w


def o4_block(l, u1, u3, u2, u4):
    """O₄ 三项和, 有序 (行 u₁,u₃ | 列 u₂,u₄), 广播向量化."""
    de = u1 + u3 - u2 - u4

    def vol(a, b, c):
        mx = np.maximum(np.maximum(a, b), np.maximum(c, 0.0))
        mn = np.minimum(np.minimum(a, b), np.minimum(c, 0.0))
        return np.clip(l - (mx - mn), 0.0, None)

    return (vol(-u1 + de, u3 - u4, -u4)
            + vol(u2 - u3 - u4, -u3 - u4, -u4)
            + vol(-u2 - u3 + u4, -u3 + u4, u4))


def direct_tables(T, prod_cap=None, chunk=800, C=None):
    """直接管线: |c| 与 signed c 按 min(N₁,N₂)/X 分箱; 返回累积表.

    C: None = 全核;  数值 = 硬截断 |N₁−N₂|·T ≤ C·min(N₁,N₂)
       (m1_suite 正式 taper 约定).
    """
    X = int(T / TWO_PI)
    l = math.log(T / TWO_PI)
    ell1 = l + 2 * LOG2 - 1
    norm = 1.0 / math.pi / (l * X * ell1 ** 4)
    # 行 = N₁ 侧全对; 列 = N₂ 侧 (可限乘积 ≤ prod_cap 只跑低 zone)
    na1, nb1, w1 = leg_pairs(X)
    na2, nb2, w2 = leg_pairs(X, prod_cap)
    N1 = na1 * nb1
    N2 = na2 * nb2
    u1, u3 = np.log(na1), np.log(nb1)
    u2, u4 = np.log(na2), np.log(nb2)
    sw1 = w1 / np.sqrt(N1)
    sw2 = w2 / np.sqrt(N2)
    edges = np.concatenate([THETA_GRID * X, [np.inf]])
    nb_ = len(edges)
    acc_abs = np.zeros(nb_)
    acc_sig = np.zeros(nb_)
    for i0 in range(0, len(N1), chunk):
        s = slice(i0, i0 + chunk)
        D = np.log(N1[s])[:, None] - np.log(N2)[None, :]
        mask = N1[s][:, None] != N2[None, :]
        if C is not None:
            mn_ = np.minimum(N1[s][:, None], N2[None, :])
            mask &= (np.abs(N1[s][:, None] - N2[None, :]) * T
                     <= C * mn_)
        with np.errstate(divide='ignore', invalid='ignore'):
            K = np.where(mask, (np.sin(2 * T * D) - np.sin(T * D))
                         / np.where(mask, D, 1.0), 0.0)
        O4 = o4_block(l, u1[s][:, None], u3[s][:, None],
                      u2[None, :], u4[None, :])
        c = sw1[s][:, None] * sw2[None, :] * K * O4
        mn = np.minimum(N1[s][:, None], np.broadcast_to(N2, (len(N1[s]),
                                                             len(N2))))
        bins = np.searchsorted(edges, mn.ravel() + 1e-9)
        acc_abs += np.bincount(bins, weights=np.abs(c).ravel(),
                               minlength=nb_)[:nb_]
        acc_sig += np.bincount(bins, weights=c.ravel(),
                               minlength=nb_)[:nb_]
    return (norm * np.cumsum(acc_abs), norm * np.cumsum(acc_sig),
            norm * acc_sig.sum(), l, X)


# ----------------------------------------------------------------------
# 均值模型
# ----------------------------------------------------------------------
_I0_Y = None
_I0_V = None


def _build_i0(ymax=3e7, npts=6_000_000):
    """I₀(y) = ∫₀^y |sin2x−sinx|/x dx 数值表 (对数网格, 一次构建)."""
    global _I0_Y, _I0_V
    if _I0_Y is not None:
        return
    y = np.concatenate([np.linspace(0, 200, 400000, endpoint=False),
                        np.exp(np.linspace(math.log(200), math.log(ymax),
                                           npts))])
    f = np.abs(np.sin(2 * y) - np.sin(y)) / np.where(y == 0, 1.0, y)
    f[0] = 1.0  # lim x→0 |sin2x−sinx|/x = 1
    v = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2
                                         * np.diff(y))])
    _I0_Y, _I0_V = y, v


def i0(y):
    """I₀ 插值 (线性; 表外用周期平均斜率 ⟨|s|⟩ 延拓)."""
    _build_i0()
    y = np.asarray(y, dtype=np.float64)
    out = np.interp(y, _I0_Y, _I0_V)
    big = y > _I0_Y[-1]
    if np.any(big):
        yy = np.linspace(0, TWO_PI, 20001)
        sbar = float(np.trapezoid(np.abs(np.sin(2 * yy) - np.sin(yy)),
                                  yy)) / TWO_PI
        out = np.where(big, _I0_V[-1] + sbar * np.log(y / _I0_Y[-1]), out)
    return out


def mean_model(T, ngrid=140, chunk=1200):
    """B_mean(Θ-网格): 4 维数值积分.

    PNT 平滑: Σ_pp Λ f(u) → ∫f e^u du, 合并 1/√(N₁N₂) 后净权
    e^{Σu/2}du⁴.  核: Δ-胞元 (半宽 du/2 划分) 内的 |Ŵin| 质量用
    I₀ 表精确差分, j≥1 格以 |Δ| ≥ 1/max(N₁,N₂) 截断模拟.
    """
    X = T / TWO_PI
    l = math.log(X)
    ell1 = l + 2 * LOG2 - 1
    norm = 1.0 / math.pi / (l * X * ell1 ** 4)
    u = np.linspace(LOG2, l, ngrid)
    du = u[1] - u[0]
    ua, ub = np.meshgrid(u, u, indexing='ij')
    ua, ub = ua.ravel(), ub.ravel()
    S = ua + ub                       # log N
    wpair = np.exp(S / 2) * du * du   # e^{(uₐ+u_b)/2} du² (每侧)
    edges = np.concatenate([np.log(THETA_GRID * X), [np.inf]])
    nb_ = len(edges)
    acc = np.zeros(nb_)
    for j0 in range(0, len(S), chunk):
        s = slice(j0, j0 + chunk)
        D = S[s][:, None] - S[None, :]
        aD = np.abs(D)
        cut = np.exp(-np.maximum(S[s][:, None], S[None, :]))  # 1/max(N)
        lo_ = np.maximum(aD - du / 2, cut)
        hi_ = np.maximum(aD + du / 2, lo_)
        mass = i0(T * hi_) - i0(T * lo_)
        # Δ=0 胞元: 两侧折叠, 质量 = 2·[I₀(T(du/2)) − I₀(T·cut)]
        mass = np.where(aD < du / 4,
                        2 * np.clip(i0(T * (aD + du / 2)) - i0(T * cut),
                                    0.0, None),
                        mass)
        Kbar = mass / du
        O4 = o4_block(l, ua[s][:, None], ub[s][:, None],
                      ua[None, :], ub[None, :])
        val = wpair[s][:, None] * wpair[None, :] * Kbar * O4
        mn = np.minimum(S[s][:, None], S[None, :])
        bins = np.searchsorted(edges, mn.ravel() + 1e-12)
        acc += np.bincount(bins, weights=val.ravel(),
                           minlength=nb_)[:nb_]
    return np.cumsum(acc) * norm


def cell_abs_tables(T, C=None, prod_cap_theta=8):
    """胞元级绝对值上界: 按 T1-胞元 (b₁,b₂,j) 聚合后取 |·|.

    b_i = 各积的最小腿 (M1 约定 b ≤ √N).  每个 Θ 先限制四元组类
    min(N₁,N₂) ≤ ΘX 再聚合 — 这是 L1′ 在 T1-胞元粒度上的正确对象:
    胞元内 m-滑动的核振荡对消是 PNT-可证的 (光滑积分), 不动用算术.
    返回 (B_cell(Θ) 数组, 量子级 B_quad(Θ) 对照数组).
    """
    X = int(T / TWO_PI)
    l = math.log(T / TWO_PI)
    ell1 = l + 2 * LOG2 - 1
    norm = 1.0 / math.pi / (l * X * ell1 ** 4)
    na1, nb1, w1 = leg_pairs(X)
    na2, nb2, w2 = leg_pairs(X, prod_cap_theta * X + 10)
    N1 = na1 * nb1
    N2 = na2 * nb2
    u1, u3 = np.log(na1), np.log(nb1)
    sw1 = w1 / np.sqrt(N1)
    b1key = np.minimum(na1, nb1).astype(np.int64)
    # 复合键 = ((b₁·512+b₂)·2²⁰ + j+2¹⁹)·8 + Θ-箱号: 单遍累积
    edges = THETA_GRID * X
    quad = np.zeros(len(THETA_GRID))
    KA, VA = [], []

    def _merge():
        k = np.concatenate(KA)
        v = np.concatenate(VA)
        uk, inv = np.unique(k, return_inverse=True)
        return [uk], [np.bincount(inv, weights=v)]

    for ci in range(len(N2)):
        D = np.log(N1) - math.log(N2[ci])
        mask = N1 != N2[ci]
        if C is not None:
            mask &= (np.abs(N1 - N2[ci]) * T
                     <= C * np.minimum(N1, N2[ci]))
        K = np.where(mask, (np.sin(2 * T * D) - np.sin(T * D))
                     / np.where(mask, D, 1.0), 0.0)
        O4 = o4_block(l, u1, u3, math.log(na2[ci]), math.log(nb2[ci]))
        c0 = sw1 * K * O4 * (w2[ci] / math.sqrt(N2[ci]))
        jj = (N1 - N2[ci]).astype(np.int64)
        b2 = int(min(na2[ci], nb2[ci]))
        mn = np.minimum(N1, N2[ci])
        sel = c0 != 0.0
        if not sel.any():
            continue
        tbin = np.searchsorted(edges, mn[sel] + 1e-9)   # 0..7 (7=Θ>8)
        aug = ((b1key[sel] * 512 + b2) * (2 ** 20)
               + (jj[sel] + 2 ** 19)) * 8 + tbin
        KA.append(aug)
        VA.append(c0[sel])
        for ti in range(len(THETA_GRID)):
            quad[ti] += float(np.abs(c0[sel][tbin <= ti]).sum())
        if len(KA) > 60:
            KA, VA = _merge()
    KA, VA = _merge()
    keys, vals = KA[0], VA[0]
    cellid = keys // 8
    tbin = keys % 8
    bc = np.zeros(len(THETA_GRID))
    for ti in range(len(THETA_GRID)):
        m = tbin <= ti
        if not m.any():
            continue
        uc, inv = np.unique(cellid[m], return_inverse=True)
        sums = np.bincount(inv, weights=vals[m])
        bc[ti] = float(np.abs(sums).sum())
    return norm * bc, norm * quad


def main():
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'direct'
    if mode == 'direct':
        for Ts in sys.argv[2:]:
            T = float(Ts)
            babs, bsig, fullR, l, X = direct_tables(T)
            print(f"T={T:.0f} (l={l:.3f}, X={X}):")
            print("  Θ:     " + "  ".join(f"{t:6.1f}" for t in THETA_GRID))
            print("  B_abs: " + "  ".join(f"{v:6.4f}"
                                          for v in babs[:len(THETA_GRID)]))
            print("  signed:" + "  ".join(f"{v:+6.4f}"
                                          for v in bsig[:len(THETA_GRID)]))
            print(f"  full-R (C=∞ 全类) = {fullR:+.5f}")
            babs40, bsig40, fullR40, _, _ = direct_tables(T, C=40.0)
            print("  [C=40] B_abs: "
                  + "  ".join(f"{v:6.4f}" for v in babs40[:len(THETA_GRID)]))
            print("  [C=40] signed:"
                  + "  ".join(f"{v:+6.4f}" for v in bsig40[:len(THETA_GRID)]))
            print(f"  [C=40] full-R = {fullR40:+.5f}")
            sys.stdout.flush()
    elif mode == 'direct-restricted':
        for Ts in sys.argv[2:]:
            T = float(Ts)
            X = int(T / TWO_PI)
            babs, bsig, _, l, _ = direct_tables(T, prod_cap=8 * X + 10)
            print(f"T={T:.0f} (l={l:.3f}) [限低zone 列侧]:")
            print("  Θ:     " + "  ".join(f"{t:6.1f}" for t in THETA_GRID))
            print("  B_abs: " + "  ".join(f"{v:6.4f}"
                                          for v in babs[:len(THETA_GRID)]))
            print("  signed:" + "  ".join(f"{v:+6.4f}"
                                          for v in bsig[:len(THETA_GRID)]))
            sys.stdout.flush()
    elif mode == 'cellabs':
        for Ts in sys.argv[2:]:
            T = float(Ts)
            bc, bq = cell_abs_tables(T)
            print(f"T={T:.0f}:")
            print("  Θ:       " + "  ".join(f"{t:6.1f}" for t in THETA_GRID))
            print("  B_cell:  " + "  ".join(f"{v:6.4f}" for v in bc))
            print("  B_quad:  " + "  ".join(f"{v:6.4f}" for v in bq))
            bc40, bq40 = cell_abs_tables(T, C=40.0)
            print("  [C=40] B_cell: " + "  ".join(f"{v:6.4f}" for v in bc40))
            sys.stdout.flush()
    elif mode == 'mean':
        for T in [600., 1200., 2400., 4800., 9600., 1e5, 1e6]:
            bm = mean_model(T)
            print(f"T={T:.0f}: B_mean(Θ) = "
                  + "  ".join(f"{v:6.4f}" for v in bm[:len(THETA_GRID)]))
            sys.stdout.flush()
    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
