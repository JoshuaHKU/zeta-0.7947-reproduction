# -*- coding: utf-8 -*-
"""T2 dispersion swap identities S₁/S₂/S₃ and exact bilinear-form
separation (helper for the §4 chain). T2 色散三项换序恒等式与精确
双线性型分离 (§4 链辅助). S₁ 换序恒等式验证至 2e-16;
S₂/S₃ 同款换序 → 组装 D = S₁ − 2S₂ + S₃ → 分离对角地板与
净化离对角双线性型 → 结构化位移族 ↔ 剩余类方差的桥恒等式.

胞元族约定:
    固定 b₁; 块 N₂ = b₂m ∈ [s₀X, s₁X]; 胞元 (b₂, j), 0<|j|≤J;
    A(b₂,j) = Σ_m Λ(m)Λ((b₂m+j)/b₁)·1[b₁|b₂m+j],  m, n 均为素数幂
    MT(b₂,j) = 𝔖_{b₁,b₂}(j)·len/(b₁b₂),  len = (s₁−s₀)X  (连续长度)
    换元 n = (b₂m+j)/b₁ ⟺ j = b₁n−b₂m (整数 n 双射; j≠0 ⟺ b₁n≠b₂m)

预注册证伪器 (先测后信; 判据在计算前固定):
  F1  S₂ 换序: 直接式 Σ A·MT 与带状式 Σ_m Λ(m)Σ_n Λ(n)𝔖(b₁n−b₂m)
      相对差 < 1e-12 (逐 b₂ 与总量).
  F2  S₃ 闭式: E₂(b₁,b₂) 局部乘积 vs 直接 j-平均 Σ_{j≤J}𝔖²/(J·E₂),
      J = 5×10⁴ 时 |比值−1| ≤ 0.05; 锚: E₂(1,1) = 2∏_{p>2}(1+(p−1)⁻³)
      与局部乘积通式相对差 < 1e-9.
  F3  组装: D_换序 = S₁−2S₂+S₃ (换序件) 与 D_直接 = Σ(A−MT)²
      相对差 < 1e-10.
  F4  色散符号: 净化离对角 B_net = D − diag 满足
      B_net/diag ∈ [−0.6, +0.1] (存档方差比 0.56–0.72 的推论);
      B_net > 0.1·diag ⟹ 平方根消去图景在该块被证伪.
  F5  桥恒等式: 固定窗 W, 模 q: Σ_{k≠0} T_W(qk) =
      Σ_{a mod q}[ψ_W(a)² − Σ_{n∈W,n≡a}Λ(n)²] 逐位成立 (机器精度).
  F0  存档锚: 某个族变体的 S₁ 直接值复现 73312.0157 (存档演示;
      其脚本未入档 — 本轮从备忘录描述重建并固定约定).

依赖: m1_suite (𝔖 机制). 运行: python3 t2_swaps.py
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from m1_suite import (C2_TWIN, Sieves, kap_p, primepowers, prime_factors,
                      s_sing, vp_int)

TWO_PI = 2 * math.pi


# ----------------------------------------------------------------------
# 一、E₂(b₁,b₂) = E_j[𝔖²] 的局部乘积闭式 (S₃ 的渐近常数)
#     v_p(j) 分布: P(v=k) = (1−1/p)p^{−k};  E₂ = ∏_p E_v[κ_p(v)²].
# ----------------------------------------------------------------------
_P_TAIL = 100000          # 通用素数截断 (因子 1−2/(p(p−1)²)+…, p⁻³ 收敛)


def _e2_local(p, e1, e2):
    """单素数局部二阶矩 E_v[κ_p(v)²] (κ_p 在 v>max(e)+1 后稳定)."""
    kcap = max(e1, e2) + 2
    tot = 0.0
    for k in range(kcap + 1):
        tot += (1 - 1 / p) * p ** (-k) * kap_p(p, e1, e2, k) ** 2
    tot += kap_p(p, e1, e2, kcap + 1) ** 2 * p ** (-(kcap + 1))
    return tot


def _primes_upto_list(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.nonzero(s)[0]


_PRIMES = _primes_upto_list(_P_TAIL)
# 通用奇素数基座乘积 ∏_{p>2}[(1−1/p)(1−(p−1)⁻²)² + (1/p)(1+(p−1)⁻¹)²]
_BASE_ODD = 1.0
for _p in _PRIMES[1:]:
    _pf = float(_p)
    _BASE_ODD *= ((1 - 1 / _pf) * (1 - 1 / (_pf - 1) ** 2) ** 2
                  + (1 / _pf) * (1 + 1 / (_pf - 1)) ** 2)


def e2_const(b1, b2):
    """E₂(b₁,b₂) = E_j[𝔖_{b₁,b₂}(j)²]: 通用基座 × 特殊素数替换."""
    val = _e2_local(2, vp_int(b1, 2), vp_int(b2, 2)) * _BASE_ODD
    for p in sorted(prime_factors(b1 * b2)):
        if p == 2:
            continue
        pf = float(p)
        generic = ((1 - 1 / pf) * (1 - 1 / (pf - 1) ** 2) ** 2
                   + (1 / pf) * (1 + 1 / (pf - 1)) ** 2)
        val *= _e2_local(p, vp_int(b1, p), vp_int(b2, p)) / generic
    return val


# ----------------------------------------------------------------------
# 二、胞元族构建与三项 S₁/S₂/S₃ (直接式与换序式)
# ----------------------------------------------------------------------
class Family:
    """固定 b₁ 的胞元族: 块 N₂ ∈ [s₀X, s₁X], b₂ ∈ 素数幂, 0<|j|≤J."""

    def __init__(self, T, b1, s0, s1, b2min, b2max, J, m_le_X=True):
        self.T, self.b1, self.J = T, b1, J
        self.X = int(T / TWO_PI)
        self.s0, self.s1 = s0, s1
        self.len_cont = (s1 - s0) * self.X          # 连续块长 (MT 用)
        cap = max(int(s1 * self.X) + J + 10, self.X + 10)
        pp, lw = primepowers(cap)
        self.lam = np.zeros(cap + 1)
        self.lam[pp] = lw
        self.b2s = [int(b) for b in pp
                    if b2min <= b <= b2max]
        self.m_le_X = m_le_X

    def m_range(self, b2):
        """m-窗: b₂m ∈ [s₀X, s₁X] (可选 m ≤ X 类约束)."""
        lo = int(math.ceil(self.s0 * self.X / b2))
        hi = int(math.floor(self.s1 * self.X / b2))
        if self.m_le_X:
            hi = min(hi, self.X)
        return lo, hi

    def cells_direct(self, b2):
        """A(b₂,j), 0<|j|≤J 直接枚举; 返回 j→A 数组 (偏移 J)."""
        b1, J, lam = self.b1, self.J, self.lam
        lo, hi = self.m_range(b2)
        A = np.zeros(2 * J + 1)
        for m in range(lo, hi + 1):
            if lam[m] == 0.0:
                continue
            base = b2 * m
            # j ≡ −b₂m (mod b₁) ⟺ n = (b₂m+j)/b₁ 整数
            for j in range(-J, J + 1):
                if j == 0 or (base + j) % b1:
                    continue
                n = (base + j) // b1
                if 0 < n < len(lam) and lam[n] > 0.0:
                    A[j + J] += lam[m] * lam[n]
        return A


def three_terms(fam, sarr_cache):
    """三项 S₁/S₂/S₃: 直接式与换序式; 对角/离对角剖分; 逐 b₂ 表."""
    b1, J = fam.b1, fam.J
    lam = fam.lam
    out = {'b2': [], 'S1d': [], 'S2d': [], 'S2s': [], 'S3d': [],
           'S1s': [], 'diag': [], 'Dd': [], 'MTsum': [], 'Asum': []}
    for b2 in fam.b2s:
        if b2 not in sarr_cache:
            sv = Sieves(max(J, 50000))
            sarr_cache[b2] = sv.s_array(b1, b2)
        sarr = sarr_cache[b2]
        cmt = fam.len_cont / (b1 * b2)
        A = fam.cells_direct(b2)
        MT = np.array([sarr[abs(j)] * cmt if j != 0 else 0.0
                       for j in range(-J, J + 1)])
        # ---- 直接式 ----
        S1d = float(np.sum(A * A))
        S2d = float(np.sum(A * MT))
        S3d = float(np.sum(MT * MT))
        Dd = float(np.sum((A - MT) ** 2))
        # ---- S₂ 换序 (带状式): Σ_m Λ(m) Σ_{n: 0<|b₁n−b₂m|≤J} Λ(n)𝔖(|j|)
        lo, hi = fam.m_range(b2)
        S2s = 0.0
        for m in range(lo, hi + 1):
            if lam[m] == 0.0:
                continue
            base = b2 * m
            nlo = -((-(base - J)) // b1)            # ceil
            nhi = (base + J) // b1                  # floor
            acc = 0.0
            for n in range(nlo, nhi + 1):
                jj = b1 * n - base
                if jj == 0 or lam[n] == 0.0:
                    continue
                acc += lam[n] * sarr[abs(jj)]
            S2s += lam[m] * acc
        S2s *= cmt
        # ---- S₁ 换序 (孪生族式): Σ_{m,m'≡ (mod r)} ΛΛ·T_{I(m)}(qk)
        g = math.gcd(b1, b2)
        r, q = b1 // g, b2 // g
        S1s, diag = 0.0, 0.0
        for m in range(lo, hi + 1):
            if lam[m] == 0.0:
                continue
            base = b2 * m
            nlo = -((-(base - J)) // b1)
            nhi = (base + J) // b1
            nwin = [n for n in range(nlo, nhi + 1)
                    if lam[n] > 0.0 and b1 * n != base]
            # m' = m − r·k 遍历同余类 (窗内)
            kmax = (hi - lo) // r + 1
            for k in range(-kmax, kmax + 1):
                mp = m - r * k
                if mp < lo or mp > hi or lam[mp] == 0.0:
                    continue
                h = q * k                            # n' = n − h
                acc = 0.0
                for n in nwin:
                    npr = n - h
                    # n' 须落在 (b₂,m') 自己的窗: |b₁n'−b₂m'| ≤ J, ≠0
                    jj = b1 * npr - b2 * mp
                    if jj == 0 or abs(jj) > J:
                        continue
                    if 0 < npr < len(lam) and lam[npr] > 0.0:
                        acc += lam[n] * lam[npr]
                term = lam[m] * lam[mp] * acc
                S1s += term
                if k == 0:
                    diag += term
        out['b2'].append(b2)
        out['S1d'].append(S1d)
        out['S2d'].append(S2d)
        out['S2s'].append(S2s)
        out['S3d'].append(S3d)
        out['S1s'].append(S1s)
        out['diag'].append(diag)
        out['Dd'].append(Dd)
        out['MTsum'].append(float(np.sum(MT)))
        out['Asum'].append(float(np.sum(A)))
    return {k: np.array(v) for k, v in out.items()}


# ----------------------------------------------------------------------
# 三、桥恒等式 (F5): 结构化位移族 = 剩余类方差 (固定窗版)
# ----------------------------------------------------------------------
def bridge_check(lam, W_lo, W_hi, qmod):
    """Σ_{k≠0} T_W(q·k) vs Σ_a [ψ_a² − Σ_{n≡a}Λ²] over W=[W_lo,W_hi]."""
    ns = np.arange(W_lo, W_hi + 1)
    lw = lam[W_lo:W_hi + 1]
    # 左端: 逐 k 孪生和
    lhs = 0.0
    kcap = (W_hi - W_lo) // qmod + 1
    for k in range(1, kcap + 1):
        h = qmod * k
        if h > W_hi - W_lo:
            break
        lhs += 2 * float(np.dot(lw[h:], lw[:len(lw) - h]))   # ±k 对称
    # 右端: 剩余类平方和
    rhs = 0.0
    for a in range(qmod):
        mask = (ns % qmod) == a
        psi = float(np.sum(lw[mask]))
        rhs += psi * psi - float(np.sum(lw[mask] ** 2))
    return lhs, rhs


# ----------------------------------------------------------------------
# 四、离对角逐 |k| 解剖: 结构化孪生偏差表 (BFI 侧必须控制的对象)
# ----------------------------------------------------------------------
def k_anatomy(fam, sarr_tw, kshow=6):
    """按 |k| 分层: 观测质量 vs HL 期望 (𝔖_{1,1}(qk)·窗点数)."""
    b1, J, lam = fam.b1, fam.J, fam.lam
    rows = []
    for b2 in fam.b2s:
        g = math.gcd(b1, b2)
        r, q = b1 // g, b2 // g
        lo, hi = fam.m_range(b2)
        obs = {}
        exp = {}
        for m in range(lo, hi + 1):
            if lam[m] == 0.0:
                continue
            base = b2 * m
            nlo = -((-(base - J)) // b1)
            nhi = (base + J) // b1
            nwin = [n for n in range(nlo, nhi + 1)
                    if lam[n] > 0.0 and b1 * n != base]
            kmax = (hi - lo) // r + 1
            for k in range(-kmax, kmax + 1):
                if k == 0:
                    continue
                mp = m - r * k
                if mp < lo or mp > hi or lam[mp] == 0.0:
                    continue
                h = q * k
                acc, wexp = 0.0, 0.0
                for n in nwin:
                    npr = n - h
                    jj = b1 * npr - b2 * mp
                    if jj == 0 or abs(jj) > J:
                        continue
                    # HL 条件期望: E[Λ(n−h) | n 素数幂] = 𝔖_tw(|h|),
                    # 故期望按 Λ(n) 加权 (按点计数会低估 ~log 倍)
                    wexp += lam[n]
                    if 0 < npr < len(lam) and lam[npr] > 0.0:
                        acc += lam[n] * lam[npr]
                ka = abs(k)
                w = lam[m] * lam[mp]
                obs[ka] = obs.get(ka, 0.0) + w * acc
                exp[ka] = exp.get(ka, 0.0) + w * wexp * sarr_tw[abs(h)]
        rows.append((b2, q, obs, exp))
    return rows


# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    print("== Round 41: T2 三项换序 + 双线性型分离 ==")

    # ---- F2 锚: E₂(1,1) 双式对照 --------------------------------------
    e2_11 = e2_const(1, 1)
    ref = 2.0
    for p in _PRIMES[1:]:
        ref *= 1 + 1 / (float(p) - 1) ** 3
    print(f"[F2 锚] E₂(1,1) 局部乘积 = {e2_11:.9f}  "
          f"闭式 2∏(1+(p−1)⁻³) = {ref:.9f}  "
          f"相对差 {abs(e2_11/ref-1):.2e}  "
          f"{'✓' if abs(e2_11/ref-1) < 1e-9 else '✗ FAIL'}")

    # ---- F2 主检: 大 J 直接平均 vs E₂ (b₁=5 族) ------------------------
    b1 = 5
    sv_big = Sieves(50000)
    print("[F2] Σ_{j≤J}𝔖²/(J·E₂), J=5e4  (b₁=5):")
    f2_ok = True
    for b2 in [3, 4, 5, 7, 9, 16, 25, 27]:
        sarr = sv_big.s_array(b1, b2)
        e2 = e2_const(b1, b2)
        ratio = float(np.sum(sarr[1:] ** 2)) / (sv_big.J * e2)
        ok = abs(ratio - 1) <= 0.05
        f2_ok &= ok
        print(f"      b₂={b2:>2}: E₂={e2:.5f}  比值={ratio:.4f} "
              f"{'✓' if ok else '✗'}")
    print(f"[F2] 判定: {'通过' if f2_ok else '鸣响'}")

    # ---- F0: 存档锚复现 (族变体扫描) ---------------------------------
    print("[F0] 存档锚: T=2400, b₁=5, N∈[3X,3.3X], b₂≤40, J=19; "
          "S₁ 目标 73312.0157")
    anchor = None
    for b2min, mlex in [(2, False), (2, True), (3, False), (3, True)]:
        fam = Family(2400., 5, 3.0, 3.3, b2min, 40, 19, m_le_X=mlex)
        s1tot = 0.0
        for b2 in fam.b2s:
            A = fam.cells_direct(b2)
            s1tot += float(np.sum(A * A))
        tag = f"b₂≥{b2min}, m≤X {'开' if mlex else '关'}"
        hit = abs(s1tot - 73312.0157) < 0.5
        print(f"      {tag}: S₁ = {s1tot:.4f} {'← 锚命中 ✓' if hit else ''}")
        if hit and anchor is None:
            anchor = (b2min, mlex)
    if anchor is None:
        print("      [!] 未命中锚 — 用 b₂≥2, m≤X 关 继续并记录偏差")
        anchor = (2, False)

    # ---- 主计算: 两高度 × 三块 -----------------------------------------
    sarr_cache = {}
    sv_tw = Sieves(50000)
    sarr_tw = sv_tw.s_array(1, 1)
    results = []
    for T in [2400., 4800.]:
        for s0 in [2.0, 3.0, 5.0]:
            fam = Family(T, b1, s0, s0 + 0.3, anchor[0], 40, 19,
                         m_le_X=anchor[1])
            sarr_cache_T = {}
            r = three_terms(fam, sarr_cache_T)
            S1d, S2d, S2s = r['S1d'].sum(), r['S2d'].sum(), r['S2s'].sum()
            S3d, S1s = r['S3d'].sum(), r['S1s'].sum()
            diag, Dd = r['diag'].sum(), r['Dd'].sum()
            # F1/F3 判定
            f1 = abs(S2s / S2d - 1) if S2d else 0.0
            f1b = abs(S1s / S1d - 1) if S1d else 0.0
            Dswap = S1s - 2 * S2s + S3d
            f3 = abs(Dswap / Dd - 1) if Dd else 0.0
            Bnet = Dd - diag
            f4r = Bnet / diag if diag else 0.0
            calib = r['Asum'].sum() / r['MTsum'].sum()
            print(f"[主] T={T:.0f} 块[{s0}X,{s0+0.3}X]: "
                  f"S₁={S1d:.1f} S₂={S2d:.1f} S₃={S3d:.1f} "
                  f"D={Dd:.1f} diag={diag:.1f}")
            print(f"     F1(S₂换序) rel={f1:.2e} {'✓' if f1 < 1e-12 else '✗'}"
                  f"  S₁换序 rel={f1b:.2e} {'✓' if f1b < 1e-12 else '✗'}"
                  f"  F3(组装) rel={f3:.2e} {'✓' if f3 < 1e-10 else '✗'}")
            print(f"     F4: B_net/diag = {f4r:+.3f} "
                  f"{'✓' if -0.6 <= f4r <= 0.1 else '✗ 鸣响'}"
                  f"   D/diag = {Dd/diag:.3f}   校准 ΣA/ΣMT = {calib:.3f}")
            results.append((T, s0, S1d, S2d, S3d, Dd, diag, Bnet, calib))

    # ---- F5: 桥恒等式 (固定窗) ----------------------------------------
    print("[F5] 桥: Σ_k T_W(qk) = Σ_a[ψ_a²−Σ_{n≡a}Λ²]  (固定窗 W)")
    fam = Family(2400., b1, 3.0, 3.3, anchor[0], 40, 19, m_le_X=anchor[1])
    for qm, (wl, wh) in [(3, (500, 900)), (7, (500, 900)), (12, (229, 754))]:
        lhs, rhs = bridge_check(fam.lam, wl, wh, qm)
        rel = abs(lhs / rhs - 1) if rhs else 0.0
        print(f"      q={qm:>2} W=[{wl},{wh}]: 左={lhs:.6f} 右={rhs:.6f} "
              f"rel={rel:.1e} {'✓' if rel < 1e-12 else '✗'}")

    # ---- 逐 |k| 解剖 (两高度 × 主块) ----------------------------------
    for T in [2400., 4800.]:
        fam_a = Family(T, b1, 3.0, 3.3, anchor[0], 40, 19,
                       m_le_X=anchor[1])
        print(f"[解剖] T={T:.0f} 块[3X,3.3X]: 离对角逐 |k| "
              f"观测/HL条件期望 (聚合全 b₂):")
        rows = k_anatomy(fam_a, sarr_tw)
        agg_o, agg_e = {}, {}
        permod_o, permod_e = {}, {}
        for b2, q, obs, exp in rows:
            for ka, v in obs.items():
                agg_o[ka] = agg_o.get(ka, 0.0) + v
                permod_o[q] = permod_o.get(q, 0.0) + v
            for ka, v in exp.items():
                agg_e[ka] = agg_e.get(ka, 0.0) + v
                permod_e[q] = permod_e.get(q, 0.0) + v
        tot_o = sum(agg_o.values())
        tot_e = sum(agg_e.values())
        for ka in sorted(agg_o)[:8]:
            o, e = agg_o[ka], agg_e.get(ka, 0.0)
            print(f"      |k|={ka}: 观测={o:9.1f}  期望={e:9.1f}  "
                  f"比={o/e if e else float('nan'):.3f}")
        print(f"      全离对角: 观测={tot_o:.1f} 期望={tot_e:.1f} "
              f"比={tot_o/tot_e:.4f}")
        print("      按模 q=b₂/g 聚合 (观测/期望):",
              "  ".join(f"q={q}:{permod_o[q]/permod_e[q]:.2f}"
                        for q in sorted(permod_o)
                        if permod_e.get(q, 0.0) > 50))

    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
