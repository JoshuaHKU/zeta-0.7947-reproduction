# -*- coding: utf-8 -*-
"""Sawtooth certification (paper §4): S₀ sharpening (grid
certification + large-K analysis) giving C_saw = 2.100, and the
C_core scaled triple integral. 锯齿认证 (论文 §4): S₀ 锐化 (网格认证
+ 大-K 解析) 给出 C_saw = 2.100, 及 C_core 缩放三重积分.

A 线 — Lemma S₀-sharp:
    K ≤ 64:  Lipschitz-网严格认证 (|S₀'| ≤ 3K; 步长 h ⟹
             max ≤ 网格max + 3K·h/2, 取 h = 2·margin/(3K)).
    K > 64:  Dirichlet 表示 S₀ = ∫_x^{2x} D̃_K − x/2 的情形界
             (备忘录); 本脚本抽样 K ∈ {128,512,4096} 数值校核.
    注册 F-S0: 认证 C_saw-cert = 网格证 max + margin ≤ 2.10.

B 线 — C_core 缩放三重积分 (CL-修正后的正确形态):
    C_core = ∫₁² dν ∫∫ dx₁ dx₂ ρ(x)-素密度 × Q̄_∞(缩放几何)
             × 𝒢-因子, 其中深 zone 𝒢 ≈ −θ[log(1/θ) + c*],
             θ = 2π e^{−(ν−1)l} 的缩放消去后化为逐-ν 的谱质量.
    实现: 直接用有限-l 大参数 (l = 60, 100) 的 zone-sum 缩放形
    数值外推 (Richardson in 1/l), 对照 core-格点趋势与梯子 −0.020(2).
    注册 F-CL: C_core-估 ∈ [−0.026, −0.014].

用法: python3 sawtooth_cert.py s0 | ccore
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from g1_ledger import qsym_exact

TWO_PI = 2 * math.pi
SI_PI = 1.8519370


def s0_net_certify(Kmax=64, margin=0.02):
    """K ≤ Kmax 的 Lipschitz-网认证: 返回认证上界."""
    worst = 0.0
    for K in range(1, Kmax + 1):
        h = 2 * margin / (3 * K)
        xs = np.arange(h / 2, TWO_PI, h)
        k = np.arange(1, K + 1)
        # 分块避免大矩阵
        mx = 0.0
        for i0 in range(0, len(xs), 4000):
            xx = xs[i0:i0 + 4000]
            vals = (np.sin(2 * np.outer(xx, k))
                    - np.sin(np.outer(xx, k))) @ (1.0 / k)
            mx = max(mx, float(np.max(np.abs(vals))))
        cert = mx + margin
        worst = max(worst, cert)
    return worst


def s0_sample_largeK():
    """K > 64 情形界的数值校核 (抽样; 证明在备忘录)."""
    out = []
    for K in [128, 512, 4096]:
        k = np.arange(1, K + 1)
        xs = np.linspace(1e-4, TWO_PI - 1e-4, 200001)
        mx = 0.0
        for i0 in range(0, len(xs), 4000):
            xx = xs[i0:i0 + 4000]
            vals = (np.sin(2 * np.outer(xx, k))
                    - np.sin(np.outer(xx, k))) @ (1.0 / k)
            mx = max(mx, float(np.max(np.abs(vals))))
        out.append((K, mx))
    return out


def ccore_scaled(l, ngrid_nu=40, ngrid_b=40, cstar=-0.55):
    """缩放三重积分的有限-l 形: zone-sum 主项在缩放变量下的求值.

    结构 (覆盖函数 cover() 同构, 加 𝒢-因子):
      ∫dν ∫∫dβ (β ∈ [max(ν−1,0)+ε, ν/2]) × Q̄(β₁,β₂,ν; l)/ℓ₁⁴-规格
      × 𝒢-质量(ν; l), 其中 𝒢-质量 = θ-薄层的 T∫𝒢θ⁻²dθ 折算:
      深 zone 谱律 𝒢 = −θ[log(1/θ)+c*] ⟹ 每单位 ν 的质量
      = −l·[(ν−1)l·log-part + c*+log2π-part] × e-权重 … 归一化后
      = −l²(ν−1) − l(c*+log 2π) 每 dν (见备忘录推导).
    素密度: 每侧 dβ·l (Mertens);  权 1/ℓ₁⁴ 与 1/(πl) 归一.
    """
    ell1 = l + 2 * math.log(2.0) - 1
    nus = np.linspace(1.0 + 1e-3, 2.0, ngrid_nu)
    dnu = nus[1] - nus[0]
    tot = 0.0
    for nu in nus:
        bmin = max(nu - 1.0, 0.0) + 1e-4
        bmax = nu / 2
        if bmax <= bmin:
            continue
        bs = np.linspace(bmin, bmax, ngrid_b)
        db = bs[1] - bs[0]
        qsum = 0.0
        for b1 in bs:
            for b2 in bs:
                qsum += qsym_exact(l, math.exp(b1 * l),
                                   math.exp((nu - b1) * l),
                                   math.exp(b2 * l),
                                   math.exp((nu - b2) * l))
        qsum *= db * db * l * l          # 素密度 dβ·l 每侧
        gmass = -(l * l * (nu - 1.0) + l * (cstar - math.log(TWO_PI)))
        tot += qsum * gmass * dnu
    return tot * TWO_PI / (math.pi * l * ell1 ** 4)


def main():
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else 's0'
    if mode == 's0':
        cert = s0_net_certify()
        print(f"[F-S0] K ≤ 64 网格认证: C ≤ {cert:.4f} "
              f"(margin 0.02 已含)  [Si(π) = {SI_PI:.4f}]")
        for K, mx in s0_sample_largeK():
            print(f"      抽样 K={K}: max|S₀| = {mx:.4f} "
                  f"{'✓ ≤ 认证带' if mx <= max(cert, 2.10) else '✗'}")
        print(f"[F-S0] 判定: C_saw-cert = max(网格, 解析大-K 2.10) = "
              f"{max(cert, 2.10):.3f}")
    elif mode == 'ccore':
        print("[F-CL] 缩放三重积分 (Richardson in 1/l):")
        vals = []
        for l in [40.0, 60.0, 100.0]:
            v = ccore_scaled(l)
            vals.append((l, v))
            print(f"      l={l:.0f}: C_core-形 = {v:+.5f}")
        # 1/l-Richardson: v(l) ≈ C + a/l
        (l1, v1), (l2, v2) = vals[-2], vals[-1]
        C = (v2 * l2 - v1 * l1) / (l2 - l1)
        ok = -0.026 <= C <= -0.014
        print(f"      外推 C_core ≈ {C:+.5f} "
              f"{'✓ (∈ [−0.026, −0.014])' if ok else '✗ 鸣响'}")
    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
