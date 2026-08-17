# -*- coding: utf-8 -*-
"""连通常数的精确有理认证 / Exact rational certification of the
connected constants (paper §11 item N1).

推广 certification/exact_t222.py 的方法到任意维、任意 ov·ov 被积函数。
Generalises the certification/exact_t222.py method to arbitrary
dimension and to the ov·ov integrand family of the connected constants.

方法 / method
------------
每一项 (P,σ) 的被积函数是**两个 overlap 因子之积**
    f(x) = (1 − spread_A(x))_+ · (1 − spread_B(x))_+ · weight(x)
其中 spread = max_i p_i − min_i p_i 取自两族位置（走步前缀 / 项掩码），
每个位置都是自由变量的单位系数线性形。

先按**序超平面** p_i − p_j = 0 剖分（这解决 max/min 的分支：在每个
序胞腔上 max、min 各由一个固定位置实现，ov 退化为单个线性式），再叠加
span = 1 的截断断点 p_i − p_j = 1。两族断点都属于同一个单位系数线性形
族 ℱ = {p_i − p_j − c : c ∈ {0,±1}}，即「只在片数、不在种类」。

由此每片上被积函数是次数 ≤ 2（{4,2} 另乘 |v| 则 ≤ 3）的多项式，逐层
积分每层最多 +1 次；闭型 Newton–Cotes 在有理节点上精确积分之。

自检 / self-check：每一片同时用「整片」和「两半」两套方案积分，
多项式次数 ≤ 规则阶数时两者必须给出**同一个有理数**；不一致即中止。
漏掉一个 kink 或阶数不足都会被这一检查抓住（与 exact_t222 同款）。

支撑 / support：0 是两族位置之一，且每个自由变量都是两个走步位置之差，
故 spread_A < 1 强制所有 |x_j| < 1 ⟹ 积分域是 [−1,1]^d。

用法 / usage:
    python3 exact_connected.py gate3        # G2: 3D 回归 t_adj/{2,2,2}
    python3 exact_connected.py phi4         # G1: Φ₄ = −1/60  (d=3, 26 项)
    python3 exact_connected.py 42 D         # {4,2} 的第 D 个放置类 (D=1,2,3)
    python3 exact_connected.py cycle B      # 纯 b-圈 (B=4,5,6)
    python3 exact_connected.py ... --terms i:j   # 只做第 i..j 项 (并行用)
    python3 exact_connected.py ... --out FILE    # 逐项精确分数落盘
"""
import itertools
import os
import sys
import time
from fractions import Fraction as F

ZERO, ONE = F(0), F(1)

# 闭型 Newton–Cotes 权重（奇数节点数，对次数 ≤ n 精确）
NC5 = [F(7, 90), F(32, 90), F(12, 90), F(32, 90), F(7, 90)]
NC7 = [F(41, 840), F(216, 840), F(27, 840), F(272, 840), F(27, 840),
       F(216, 840), F(41, 840)]
NC9 = [F(989, 28350), F(5888, 28350), F(-928, 28350), F(10496, 28350),
       F(-4540, 28350), F(10496, 28350), F(-928, 28350), F(5888, 28350),
       F(989, 28350)]


def nc_int(g, x0, x1, wts):
    n = len(wts) - 1
    h = x1 - x0
    s = ZERO
    for i, wt in enumerate(wts):
        s += wt * g(x0 + h * F(i, n))
    return s * h


def nc_checked(g, x0, x1, wts, tag):
    """整片 vs 两半：次数 ≤ 阶数时必须逐位相同。"""
    a = nc_int(g, x0, x1, wts)
    xm = (x0 + x1) / 2
    b = nc_int(g, x0, xm, wts) + nc_int(g, xm, x1, wts)
    if a != b:
        raise RuntimeError(f"scheme disagreement [{x0},{x1}] @{tag}")
    return a


# ---------------------------------------------------------------------
# 仿射形：(c0, c1, …, cd) 表示 c0 + Σ c_i x_i
# ---------------------------------------------------------------------
def aff_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def aff_eval(a, xs):
    """xs 给出前 len(xs) 个变量的值；返回 (常数部分, 剩余系数)。"""
    c = a[0]
    for i, v in enumerate(xs):
        c += a[i + 1] * v
    return c


def solve_for(a, k):
    """把仿射形 a（在变量 0..k 上）解出 x_k，返回关于 x_0..x_{k-1} 的仿射形；
    若 x_k 系数为 0 返回 None。"""
    ck = a[k + 1]
    if ck == 0:
        return None
    out = [-a[0] / ck]
    for i in range(k):
        out.append(-a[i + 1] / ck)
    return tuple(out)


def pad(a, d):
    return tuple(list(a) + [ZERO] * (d + 1 - len(a)))


class Cls:
    """一个 (走步A, 位置族B, 权重) 的精确积分任务。"""

    def __init__(self, d, posA, posB, wsign=None, name=""):
        self.d = d
        self.posA = [pad(p, d) for p in posA]
        self.posB = [pad(p, d) for p in posB]
        self.wsign = wsign          # None 或 变量下标 k：权重 = |x_k|
        self.name = name
        self.forms = self._forms()
        self.BP = self._breakpoints()

    # ---- kink 线性形族 ℱ ----
    def _forms(self):
        d = self.d
        fs = set()
        for pos in (self.posA, self.posB):
            for i in range(len(pos)):
                for j in range(len(pos)):
                    if i == j:
                        continue
                    dd = aff_sub(pos[i], pos[j])
                    if not any(dd):
                        continue
                    for c in (F(-1), ZERO, ONE):
                        fs.add(tuple([dd[0] - c] + list(dd[1:])))
        if self.wsign is not None:      # |x_k| 在 x_k = 0 有 kink
            fs.add(tuple([ZERO] + [ONE if i == self.wsign else ZERO
                                   for i in range(d)]))
        return sorted(fs)

    # ---- 逐层断点候选 ----
    def _breakpoints(self):
        d = self.d
        BP = [None] * d
        # 最内层 x_{d-1}
        cur = set()
        for L in self.forms:
            s = solve_for(L, d - 1)
            if s is not None:
                cur.add(s)
        BP[d - 1] = sorted(cur)
        # 向外传播
        for k in range(d - 2, -1, -1):
            cand = set()
            # (iii) 不含更内层变量的直接 kink
            for L in self.forms:
                if any(L[i + 1] != 0 for i in range(k + 1, d)):
                    continue
                s = solve_for(L, k)
                if s is not None:
                    cand.add(s)
            inner = BP[k + 1]
            # (ii) 内层断点撞边界 ±1
            for f in inner:
                for e in (F(-1), ONE):
                    L = tuple([f[0] - e] + list(f[1:]))
                    s = solve_for(pad(L, d), k)
                    if s is not None:
                        cand.add(s)
            # (i) 两个内层断点相撞
            for a in range(len(inner)):
                for b in range(a + 1, len(inner)):
                    L = aff_sub(pad(inner[a], d), pad(inner[b], d))
                    s = solve_for(L, k)
                    if s is not None:
                        cand.add(s)
            BP[k] = sorted(cand)
        return BP

    # ---- 被积函数 ----
    def f(self, xs):
        va = [aff_eval(p, xs) for p in self.posA]
        sa = max(va) - min(va)
        if sa >= 1:
            return ZERO
        vb = [aff_eval(p, xs) for p in self.posB]
        sb = max(vb) - min(vb)
        if sb >= 1:
            return ZERO
        r = (1 - sa) * (1 - sb)
        if self.wsign is not None:
            w = abs(xs[self.wsign])
            r *= (w if w < 1 else ONE)
        return r

    # ---- 逐层精确积分 ----
    def _rule(self, k):
        # 该层被积函数次数 ≤ 2(+1) + (d-1-k)；给足余量
        deg = 3 + (self.d - 1 - k)
        return NC5 if deg <= 5 else (NC7 if deg <= 7 else NC9)

    def integrate(self, k=0, xs=()):
        if k == self.d:
            return self.f(xs)
        bs = {F(-1), ONE}
        for bp in self.BP[k]:
            v = aff_eval(bp, xs)
            if -1 <= v <= 1:
                bs.add(v)
        bs = sorted(bs)
        wts = self._rule(k)
        tot = ZERO
        for x0, x1 in zip(bs, bs[1:]):
            if x1 <= x0:
                continue
            tot += nc_checked(lambda t: self.integrate(k + 1, xs + (t,)),
                              x0, x1, wts, f"{self.name}/L{k}")
        return tot

    def cost(self):
        return [len(b) for b in self.BP]


# ---------------------------------------------------------------------
# 项族构造 / term families
# ---------------------------------------------------------------------
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
    out = []
    for P in set_partitions(list(range(b))):
        m = len(P)
        sign = (-1) ** (m - 1)
        first = next(i for i, blk in enumerate(P) if 0 in blk)
        others = [i for i in range(m) if i != first]
        for perm in itertools.permutations(others):
            order = [first] + list(perm)
            masks, cur = [], frozenset()
            for bi in order[:-1]:
                cur = cur | frozenset(P[bi])
                masks.append(tuple(sorted(cur)))
            out.append((sign, tuple(masks)))
    return out


def cycle_terms(b):
    """纯 b-圈：自由变量 v_0..v_{b-2}，v_{b-1} = −Σ。"""
    d = b - 1
    walk = []
    for k in range(b):                       # 0, v0, v0+v1, …
        walk.append(tuple([ZERO] + [ONE if j < k else ZERO for j in range(d)]))
    out = []
    for sign, masks in compile_terms(b):
        pos = [tuple([ZERO] * (d + 1))]
        for mk in masks:
            co = [ZERO] * (d + 1)
            for j in mk:
                if j < d:
                    co[j + 1] += ONE
                else:
                    for i in range(d):
                        co[i + 1] -= ONE
            pos.append(tuple(co))
        out.append((sign, Cls(d, walk, pos, None, f"cyc{b}")))
    return out


def spec42_terms(dcls):
    """{4,2} 第 dcls 个放置类；自由变量 (v,c1,c2,c3) = x0..x3。"""
    d = 4
    V  = (ZERO, ONE, ZERO, ZERO, ZERO)
    S1 = (ZERO, ZERO, ONE, ZERO, ZERO)
    S2 = (ZERO, ZERO, ONE, ONE, ZERO)
    S3 = (ZERO, ZERO, ONE, ONE, ONE)
    Z  = (ZERO, ZERO, ZERO, ZERO, ZERO)
    add = lambda a, b: tuple(x + y for x, y in zip(a, b))
    if dcls == 1:
        walk = [Z, V, Z, S1, S2, S3]
    elif dcls == 2:
        walk = [Z, V, add(V, S1), S1, S2, S3]
    else:
        walk = [Z, V, add(V, S1), add(V, S2), S2, S3]
    W = [S1,                                        # c1
         aff_sub(S2, S1),                           # c2
         aff_sub(S3, S2),                           # c3
         tuple(-x for x in S3)]                     # c4 = -(c1+c2+c3)
    out = []
    for sign, masks in compile_terms(4):
        pos = [Z]
        for mk in masks:
            co = Z
            for j in mk:
                co = add(co, W[j])
            pos.append(co)
        out.append((sign, Cls(d, walk, pos, 0, f"42d{dcls}")))
    return out


# ---------------------------------------------------------------------
if __name__ == "__main__":
    what = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    lo, hi, out = 0, 10 ** 9, None
    if "--terms" in sys.argv:
        s = sys.argv[sys.argv.index("--terms") + 1]
        lo, hi = (int(x) for x in s.split(":"))
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    if what == "phi4":
        terms, target = cycle_terms(4), F(-1, 60)
    elif what == "cycle":
        b = int(args[0])
        terms = cycle_terms(b)
        target = {4: F(-1, 60), 5: F(1, 36), 6: F(-1, 126)}.get(b)
    elif what == "42":
        dcls = int(args[0])
        terms = spec42_terms(dcls)
        target = {1: F(-2, 315), 2: F(-1, 1260), 3: F(-1, 252)}[dcls]
    else:
        print(__doc__); sys.exit(1)

    hi = min(hi, len(terms))
    print(f"# {what} {args}: {len(terms)} terms, doing [{lo},{hi})", flush=True)
    if terms:
        print(f"# breakpoint counts per level (term 0): {terms[0][1].cost()}",
              flush=True)
    tot = ZERO
    t0 = time.time()
    fh = open(out, "a") if out else None
    for i in range(lo, hi):
        sign, cls = terms[i]
        ts = time.time()
        val = cls.integrate()
        tot += sign * val
        line = (f"{what} {'.'.join(map(str,args))} term {i} sign {sign} "
                f"value {val} wall {time.time()-ts:.2f}")
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()
    if fh:
        fh.close()
    print(f"# partial total [{lo},{hi}) = {tot} = {float(tot):.15f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    if target is not None and lo == 0 and hi == len(terms):
        print(f"# target {target} = {float(target):.15f}  "
              f"{'MATCH' if tot == target else 'MISMATCH'}", flush=True)
