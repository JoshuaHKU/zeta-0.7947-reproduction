# -*- coding: utf-8 -*-
"""Fourth-moment constant suite (paper §4): self-contained
reproduction of the singular-series local factors, the vector sieve,
the {2,2} engine anchors, the core deviation and the alpha-truncation.
第四矩常数套件 (论文 §4): 奇异级数局部因子、向量筛、{2,2} 引擎锚、
core 偏差与 α-截断的自含复现, 供独立审计.

约定与记号 (λ=1):

    l   = log(T/2π),  X = ⌊T/2π⌋,  ℓ₁ = l + 2·log2 − 1
    归一化:  R-单位 = (1/π) · Σ° / (d·ℓ₁⁴),  d = l·X
    核:      Ŵin(Δ) = [sin(2TΔ) − sin(TΔ)]/Δ,  Δ = log(N₁/N₂)
    C-taper 【本套件的正式约定】: 硬截断 |N₁−N₂|·T ≤ C·min(N₁,N₂),
             即 j-级数截于 |j| ≤ C/θ (θ = T/N); 本套件以此硬截断为唯一约定.

主公式 (round 32 推导, round 33 修正枚举):

    R_pp^模型 = (1/π d ℓ₁⁴) · Σ_{b₁≤b₂ 素数幂} Λ(b₁)Λ(b₂)/(b₁b₂)
                · T ∫ 𝒢_{b₁,b₂}(θ) · Q_sym(β₁,β₂,ω) · θ⁻² dθ

    𝒢(θ)   = Σ_{j≠0} D(j)·[sin(2θj) − sin(θj)]/j       (剪切密度级数)
    D(j)   = 𝔖_{b₁,b₂}(j)   Hardy–Littlewood 精确局部乘积 (算术模型)
           = g·1_{g|j}       整数分拆零模型 (g = gcd)
    Q_sym  = Q(β₁,β₂) + Q(β₂,β₁)  [O₄ 三项和对行/列不对称 — round 33 修正]

依赖: numpy. 运行 `python3 m1_suite.py validate` 复现全部关键数字.
"""
import math
import sys
import time

import numpy as np

TWO_PI = 2 * math.pi
LOG2 = math.log(2.0)
# 孪生素数常数 C₂ = ∏_{p>2} (1 − (p−1)⁻²)
C2_TWIN = 0.6601618158468696


# ----------------------------------------------------------------------
# 一、基础数论工具
# ----------------------------------------------------------------------
def primes_upto(n):
    """埃氏筛: 返回 ≤ n 的全部素数 (ndarray)."""
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.nonzero(s)[0]


def primepowers(x):
    """素数幂表: 返回 (n 数组, Λ(n) 数组), n = p^k ≤ x, Λ(n) = log p."""
    lam = {}
    for p in range(2, x + 1):
        if all(p % q for q in range(2, int(p ** 0.5) + 1)):
            pk = p
            while pk <= x:
                lam[pk] = math.log(p)
                pk *= p
    ns = np.array(sorted(lam), dtype=np.int64)
    return ns, np.array([lam[n] for n in ns])


def vp_int(n, p):
    """p-adic 赋值 v_p(n)."""
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def prime_factors(n):
    """n 的素因子集合."""
    fs, d = set(), 2
    while d * d <= n:
        if n % d == 0:
            fs.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


# ----------------------------------------------------------------------
# 二、奇异级数 𝔖_{b₁,b₂}(j) —— 线性型 b₁m₁ − b₂m₂ = j 的 HL 局部密度
#     归一化: E_j[𝔖] = 1;  每单位 N、每单位 j 的解密度 = 𝔖/(b₁b₂).
#     round 32 推导闭式局部因子, 对残差暴力计数逐位验证.
# ----------------------------------------------------------------------
def kap_p(p, e1, e2, v):
    """局部因子 κ_p.  e₁=v_p(b₁), e₂=v_p(b₂), v=v_p(j), e=min(e₁,e₂).

    情形表 (round 32, 逐位验证):
      e₁=e₂=0        : p|j → 1+1/(p−1) ;  否则 1−(p−1)⁻²      [孪生型]
      恰一个 eᵢ>0    : p|j → 0 ;          否则 p/(p−1)
      e₁=e₂=e≥1      : v<e → 0 ;  v=e → p^e(1−(p−1)⁻²) ;
                       v>e → p^e(1+1/(p−1))
      e₁≠e₂ 皆≥1     : v=e → p^e·p/(p−1) ;  否则 0
    """
    if e1 == 0 and e2 == 0:
        return 1 + 1 / (p - 1) if v > 0 else 1 - 1 / (p - 1) ** 2
    if min(e1, e2) == 0:
        return 0.0 if v > 0 else p / (p - 1)
    e = min(e1, e2)
    if e1 == e2:
        if v < e:
            return 0.0
        if v == e:
            return p ** e * (1 - 1 / (p - 1) ** 2)
        return p ** e * (1 + 1 / (p - 1))
    return p ** e * p / (p - 1) if v == e else 0.0


def s_sing(b1, b2, j):
    """𝔖_{b₁,b₂}(j) 标量版: κ₂ · C₂ · ∏_{p 奇, p|jb₁b₂} κ_p/(1−(p−1)⁻²)."""
    aj = abs(j)
    k2 = kap_p(2, vp_int(b1, 2), vp_int(b2, 2), vp_int(aj, 2))
    if k2 == 0.0:
        return 0.0
    val = k2 * C2_TWIN
    for p in sorted(prime_factors(b1 * b2 * aj)):
        if p == 2:
            continue
        kp = kap_p(p, vp_int(b1, p), vp_int(b2, p), vp_int(aj, p))
        if kp == 0.0:
            return 0.0
        val *= kp / (1 - 1 / (p - 1) ** 2)
    return val


def s_brute(b1, b2, j, plist=(2, 3, 5, 7, 11, 13)):
    """暴力验证: 对小素数 p 直接在 (ℤ/p^s)² 中数单位解, 与闭式对照."""
    val = 1.0
    for p in plist:
        s = vp_int(b1, p) + vp_int(b2, p) + vp_int(abs(j), p) + 2
        ps = p ** s
        cnt = 0
        for x in range(ps):
            if x % p == 0:
                continue
            for y in range(ps):
                if y % p == 0:
                    continue
                if (b1 * x - b2 * y - j) % ps == 0:
                    cnt += 1
        val *= cnt / (ps * (1 - 1 / p) ** 2)
    ref = 1.0
    for p in plist:
        ref *= kap_p(p, vp_int(b1, p), vp_int(b2, p), vp_int(abs(j), p))
    return val, ref


class Sieves:
    """j-数组筛: TW(j) = ∏_{奇 p|j}(p−1)/(p−2) 与 v₂(j), v_p(j) 表.

    用途: 把 𝔖_{b₁,b₂}(j) 向量化到整段 j = 1..J (round 33; vparr 的
    覆盖式写法修正了奇素数多剩余类的 bug).
    """

    def __init__(self, jmax):
        self.J = jmax
        tw = np.ones(jmax + 1)
        for p in primes_upto(jmax):
            if p == 2:
                continue
            tw[p::p] *= (p - 1) / (p - 2)
        self.TW = tw
        v2 = np.zeros(jmax + 1, dtype=np.int8)
        step, v = 2, 1
        while step <= jmax:
            v2[step::2 * step] = v          # 2 的各幂剩余类恰为单类
            step *= 2
            v += 1
        self.v2 = v2

    def vparr(self, p):
        """v_p(j) 数组 —— 升幂覆盖写法 (奇素数每层有 p−1 个剩余类)."""
        vp = np.zeros(self.J + 1, dtype=np.int8)
        step, v = p, 1
        while step <= self.J:
            vp[step::step] = v              # 先粗写, 高层覆盖
            step *= p
            v += 1
        return vp

    def s_array(self, b1, b2):
        """𝔖_{b₁,b₂}(j), j = 0..J 向量化 (对 s_sing 零失配验证)."""
        J = self.J
        e1_2, e2_2 = vp_int(b1, 2), vp_int(b2, 2)
        v2max = int(self.v2.max()) + 2
        k2tab = np.array([kap_p(2, e1_2, e2_2, v) for v in range(v2max + 1)])
        s = k2tab[self.v2[:J + 1]] * C2_TWIN * self.TW[:J + 1].copy()
        for p in sorted(prime_factors(b1 * b2)):
            if p == 2:
                continue
            e1, e2 = vp_int(b1, p), vp_int(b2, p)
            vparr = self.vparr(p)[:J + 1]
            vmax = int(vparr.max()) + 1
            # 通用因子 (已含于 C₂·TW) 换成局部 κ_p
            gen = np.array([(1 - 1 / (p - 1) ** 2)
                            * ((p - 1) / (p - 2) if v > 0 else 1.0)
                            for v in range(vmax + 1)])
            loc = np.array([kap_p(p, e1, e2, v) for v in range(vmax + 1)])
            ratio = np.where(gen > 0, loc / np.where(gen > 0, gen, 1.0), 0.0)
            s *= ratio[vparr]
        s[0] = 0.0
        return s


# ----------------------------------------------------------------------
# 三、O₄ 几何 —— R-类测量管线的三项重叠体积 (证认代码的精确转写)
#     注意: 三项和对 行(N₁)/列(N₂) 交换【不对称】 (round 33 发现),
#     因此无序 (b₁,b₂) 胞元的权重必须取 Q(β₁,β₂)+Q(β₂,β₁).
# ----------------------------------------------------------------------
def _o4_single(L, a, b, c):
    """单个重叠体积 (L − (max{0,a,b,c} − min{0,a,b,c}))₊."""
    mx = max(0.0, a, b, c)
    mn = min(0.0, a, b, c)
    return max(L - (mx - mn), 0.0)


def o4_triple(L, u1, u3, u2, u4):
    """管线的三项 O₄ 之和; (u1,u3)=行积的两腿, (u2,u4)=列积的两腿."""
    de = u1 + u3 - u2 - u4
    return (_o4_single(L, -u1 + de, u3 - u4, -u4)
            + _o4_single(L, u2 - u3 - u4, -u3 - u4, -u4)
            + _o4_single(L, -u2 - u3 + u4, -u3 + u4, u4))


def q4(L, be1, m1, be2, m2):
    """胞元 (b₁,b₂) 的 4 腿次序和: (b,m)×(b,m) 四种位置分配."""
    tot = 0.0
    for a1, a3 in ((be1, m1), (m1, be1)):
        for a2, a4 in ((be2, m2), (m2, be2)):
            tot += o4_triple(L, a1, a3, a2, a4)
    return tot


# ----------------------------------------------------------------------
# 四、zone-求和求值器 (round 33) —— R_pp 模型在给定高度的精确求值
# ----------------------------------------------------------------------
def g0_sawtooth(x):
    """G₀(x) = 2Σ_{h≥1}[sin(2xh) − sin(xh)]/h 的锯齿闭式 (向量化)."""
    def saw(y):
        ym = np.mod(y, TWO_PI)
        return np.where(ym == 0, 0.0, (math.pi - ym) / 2)
    return 2 * (saw(2 * x) - saw(x))


def dbar_alpha(b1, b2, vcap=10):
    """局部均值 D̄(j) = Σ_c α_c·1_{c|j} 的有限 Möbius 分解.

    D̄ = κ₂(v₂(j))·∏_{p 奇|b₁b₂}κ_p(v_p(j)),  E_j[D̄] = 1 (精确).
    返回 [(c, α_c)] 列表; 锯齿闭式: 𝒢_mean = Σ_c (α_c/c)·G₀(cθ).
    """
    tabs = []
    for p in sorted(prime_factors(b1 * b2) | {2}):
        e1, e2 = vp_int(b1, p), vp_int(b2, p)
        kv = [kap_p(p, e1, e2, v) for v in range(vcap + 1)]
        gam = [kv[0]] + [kv[a] - kv[a - 1] for a in range(1, vcap + 1)]
        tabs.append((p, gam))
    combos = [(1, 1.0)]
    for p, gam in tabs:
        combos = [(c * p ** a, al * g)
                  for c, al in combos for a, g in enumerate(gam) if g != 0.0]
    return combos


class ZoneSum:
    """全 θ-域 zone-求和: R_pp 模型 (可选 e-cut / C-taper / α-cut).

    参数:
      T      : 高度 (X = ⌊T/2π⌋)
      C      : None=全核;  数值=硬截断 |j| ≤ C/θ (正式 taper 约定)
      ecut   : None=全类;  数值 e_c = 边缘割 max-腿 ≥ l − e_c
      alpha1 : True 时用 α≤1 截断核行 (t-积分止于 2π/Δ; 审计用)
    """

    def __init__(self, T, C=None, ecut=None, alpha1=False,
                 thmax=200.0, pp_cap=None):
        self.T, self.C, self.ecut, self.alpha1 = T, C, ecut, alpha1
        self.l = math.log(T / TWO_PI)
        self.X = int(T / TWO_PI)
        self.ell1 = self.l + 2 * LOG2 - 1
        self.norm4 = self.l * self.X * self.ell1 ** 4     # d·ℓ₁⁴
        self.thmin = 4 * math.pi ** 2 / T                 # N = X² 端
        self.thmax = thmax if C is None else min(thmax, C)
        osc = C if C is not None else 200.0               # 振荡分辨要求
        npts = int(math.log(self.thmax / self.thmin) * max(120, int(3.2 * osc))) + 2
        self.th = np.exp(np.linspace(math.log(self.thmin),
                                     math.log(self.thmax), npts))
        self.om = np.log(TWO_PI / self.th)                # ω = log(N/X)
        jcap = (int(C * T / (4 * math.pi ** 2)) + 2 if C is not None
                else int(200 / self.thmin) + 2)
        self.sv = Sieves(min(jcap, 60000))
        cap = self.X if pp_cap is None else min(self.X, int(pp_cap))
        self.pp, self.lw = primepowers(cap)

    # -- Q(β₁,β₂,ω) 向量化 (含 e-cut) --------------------------------
    def qvec(self, be1, be2):
        l, om = self.l, self.om
        m1, m2 = l + om - be1, l + om - be2

        def o4v(a, b, c):
            z = np.zeros_like(a)
            mx = np.maximum.reduce([z, a, b, c])
            mn = np.minimum.reduce([z, a, b, c])
            return np.clip(l - (mx - mn), 0, None)

        tot = np.zeros_like(om)
        for u1, u3 in ((be1 + 0 * om, m1), (m1, be1 + 0 * om)):
            for u2, u4 in ((be2 + 0 * om, m2), (m2, be2 + 0 * om)):
                de = u1 + u3 - u2 - u4
                tot += (o4v(-u1 + de, u3 - u4, -u4)
                        + o4v(u2 - u3 - u4, -u3 - u4, -u4)
                        + o4v(-u2 - u3 + u4, -u3 + u4, u4))
        if self.ecut is not None:
            tot = np.where(np.minimum(be1, be2) <= om + self.ecut, tot, 0.0)
        return tot

    # -- 𝒢 级数 (含三种核行变体) --------------------------------------
    def _rows(self, th, J):
        """核行 [·]/j: 全核 sin(2θj)−sin(θj);  α-cut 时上端 sin 截零."""
        j = np.arange(1, J + 1)
        if not self.alpha1:
            return (np.sin(2 * th * j) - np.sin(th * j)) / j
        x = th * j
        up = np.where(x < math.pi, np.sin(2 * x), 0.0)
        return np.where(x < TWO_PI, (up - np.sin(x)) / j, 0.0)

    def gfull(self, b1, b2, sel):
        th = self.th[sel]
        if self.C is not None or self.alpha1:
            s = self.sv.s_array(b1, b2)
            g = np.zeros(len(th))
            for i, t in enumerate(th):
                J = min(int((self.C if self.C is not None else TWO_PI) / t) + 1,
                        self.sv.J)
                if J >= 1:
                    g[i] = 2 * float(np.dot(s[1:J + 1], self._rows(t, J)))
            return g
        # C=∞: 均值锯齿闭式 + 快收敛涨落级数 (J = 200/θ, cap 已验稳)
        combos = dbar_alpha(b1, b2)
        g = np.zeros(len(th))
        for c, al in combos:
            g += (al / c) * g0_sawtooth(c * th)
        s = self.sv.s_array(b1, b2)
        db = np.zeros(self.sv.J + 1)
        for c, al in combos:
            db[c::c] += al
        f = s - db
        f[0] = 0.0
        for i, t in enumerate(th):
            J = min(int(200 / t), self.sv.J)
            if J >= 1:
                g[i] += 2 * float(np.dot(f[1:J + 1], self._rows(t, J)))
        return g

    # -- 主循环 -------------------------------------------------------
    def run(self, bpairs=None, zone_bins=12):
        T, X = self.T, self.X
        pp, lw = self.pp, self.lw
        if bpairs is None:
            bpairs = [(int(pp[i]), int(pp[k]))
                      for i in range(len(pp)) for k in range(i, len(pp))]
        tot = 0.0
        zones = np.zeros(zone_bins + 1)
        for b1, b2 in bpairs:
            w = (lw[np.searchsorted(pp, b1)]
                 * lw[np.searchsorted(pp, b2)] / (b1 * b2))
            bmn, bmx = min(b1, b2), max(b1, b2)
            th = self.th
            sel = th <= T / bmx ** 2                       # b ≤ √N
            sel &= (bmn >= TWO_PI / th) | (th >= TWO_PI)   # m ≤ X (N>X 时)
            if self.ecut is not None:
                sel &= math.log(bmn) <= self.om + self.ecut
            if not sel.any():
                continue
            # round-33 修正: 无序胞元权重 = 两个有序 Q 之和
            q = self.qvec(math.log(b1), math.log(b2))[sel]
            if b1 != b2:
                q = q + self.qvec(math.log(b2), math.log(b1))[sel]
            if not q.any():
                continue
            g = self.gfull(b1, b2, sel)
            f = g * q / self.th[sel] ** 2
            tot += w * T * np.trapezoid(f, self.th[sel])
            # zone 分箱 (N = T/θ; 近似按被积点归箱)
            nv = T / self.th[sel]
            zb = np.minimum((nv // X).astype(int), zone_bins)
            dth = np.zeros(len(f))
            if len(f) > 1:
                dth[0] = (self.th[sel][1] - self.th[sel][0]) / 2
                dth[-1] = (self.th[sel][-1] - self.th[sel][-2]) / 2
                dth[1:-1] = (self.th[sel][2:] - self.th[sel][:-2]) / 2
            for zi in range(zone_bins + 1):
                m = zb == zi
                if m.any():
                    zones[zi] += w * T * float(np.sum(f[m] * dth[m]))
        return tot / math.pi / self.norm4, zones / math.pi / self.norm4


# ----------------------------------------------------------------------
# 五、费米子引擎复现 (round 18 → round 35 审计)
#     结论 (round 35): R_pp-engine = 0.2677 − 17/60 = −0.0156 是
#     【跨约定之差】(引擎 α-几何全 {2,2} 减 管线位置-几何对角记账),
#     不是单一约定下的类值; 素数侧单约定值为 −0.020(2).
# ----------------------------------------------------------------------
def engine_22():
    """复现引擎 {2,2}: C₂(α) = min(|α|,1) (支撑 |α|≤1 — GUE 三角形).

    t_adj (τ-相邻配对, ×2) 与 t_opp (对面配对); 参考值 0.11717/0.03333.
    """
    du = 0.002
    u = np.arange(-1, 1 + du / 2, du)
    c2 = np.minimum(np.abs(u), 1.0)
    tt = np.arange(-0.5, 0.5 + du / 2, du)
    gfun = np.array([np.sum(c2[(u >= -0.5 - t) & (u <= 0.5 - t)] * du)
                     for t in tt])
    t_adj = float(np.sum(gfun ** 2) * du)
    dv = 0.004
    vv = np.arange(-1, 1 + dv / 2, dv)
    va, vb = np.meshgrid(vv, vv, indexing='ij')
    qs = [np.zeros_like(va), va, va + vb, vb]
    mx = np.maximum.reduce(qs)
    mn = np.minimum.reduce(qs)
    o4 = np.clip(1.0 - (mx - mn), 0.0, None)
    t_opp = float(np.sum(o4 * np.minimum(np.abs(va), 1)
                         * np.minimum(np.abs(vb), 1)) * dv * dv)
    return t_adj, t_opp


# ----------------------------------------------------------------------
# 六、验证主程序 —— 复现论文 §4 的关键常数
# ----------------------------------------------------------------------
def validate():
    t0 = time.time()
    print("== m1_suite 验证 (关键数字复现) ==")

    # (1) 奇异级数: 闭式 vs 暴力残差计数
    bad = 0
    for b1, b2, j in [(2, 3, 1), (2, 2, 4), (3, 5, 2), (2, 4, 2),
                      (4, 6, 10), (3, 9, 6)]:
        bv, ref = s_brute(b1, b2, j)
        if abs(bv - ref) > 1e-9 * max(1, abs(ref)):
            bad += 1
    print(f"(1) 𝔖 局部因子 闭式=暴力: {6-bad}/6 通过")

    # (2) 向量筛 vs 标量
    sv = Sieves(20000)
    import random
    random.seed(7)
    bad = 0
    for _ in range(100):
        b1 = random.choice([2, 3, 4, 5, 7, 9, 12, 16, 27, 49])
        b2 = random.choice([2, 3, 4, 5, 7, 9, 12, 16, 27, 49])
        j = random.randint(1, 20000)
        if abs(sv.s_array(b1, b2)[j] - s_sing(b1, b2, j)) > 1e-10:
            bad += 1
    print(f"(2) 𝔖 向量筛 vs 标量: {100-bad}/100 通过")

    # (3) 引擎复现
    t_adj, t_opp = engine_22()
    print(f"(3) 引擎 {{2,2}}: t_adj={t_adj:.5f} (参考 .11717)  "
          f"t_opp={t_opp:.5f} (参考 .03333)  "
          f"跨约定差 = {2*t_adj+t_opp-17/60:+.5f} (−0.0156)")

    # (4) zone-求和 T=2400: e-cut 与无 e-cut (round 33 表)
    zs = ZoneSum(2400., C=None, ecut=1.5)
    tot, zones = zs.run()
    print(f"(4) T=2400 C=∞ e-cut: {tot:+.5f} (参考 −0.00518; 实测 EDGE −0.00556)")
    print("    zone 0..4:", " ".join(f"{v:+.5f}" for v in zones[:5]),
          " (实测 −0.00043/+0.00121/−0.00057/+0.00059/−0.00087)")
    zs = ZoneSum(2400., C=None, ecut=None)
    tot2, _ = zs.run()
    print(f"    无 e-cut: {tot2:+.5f} (参考 −0.00544)")

    # (5) α≤1 截断 (round 35 审计判别)
    zs = ZoneSum(2400., C=None, ecut=None, alpha1=True)
    tot3, _ = zs.run()
    print(f"(5) α≤1 截断: {tot3:+.5f} (参考 −0.00490; α>1 份额小 ⟹ "
          f"张力源于跨约定记账, 非 α-窗口)")

    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
    else:
        print(__doc__)
