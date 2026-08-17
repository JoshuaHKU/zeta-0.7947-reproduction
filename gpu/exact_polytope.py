# -*- coding: utf-8 -*-
"""连通常数的精确认证 —— 多面体路线 / polytope route.

把每一项的 ov·ov 积分**提升**成一个多面体的体积，从而把成本从
「每层断点数的连乘」换成「面格大小」。

    ov(P) = (1 − (max_i p_i − min_i p_i))_+ = Leb{ t : t ≤ p_i ≤ t+1 ∀i }

故
    ∫_{R^d} ov_A(x)·ov_B(x) dx
      = Vol_{d+2}{ (x,t,s) : t ≤ A_i(x) ≤ t+1,  s ≤ B_j(x) ≤ s+1 }

因为 0 同时属于两族位置，t,s ∈ [−1,0]，多面体有界。全部系数在
{0,±1}，右端在 {0,1}。

带权情形（{4,2} 的 min(|v|,1) = |v|，支撑内 |v|<1）再提升一次：
    ∫_P |z_k| dz = Vol_{n+1}{(z,y) : z∈P, z_k≥0, 0≤y≤z_k}
                 + Vol_{n+1}{(z,y) : z∈P, z_k≤0, 0≤y≤−z_k}

体积用有理递归（对面做递推，投影掉一个坐标，‖a‖ 的无理数正好约掉）：

    Vol_n(P) = (1/n) Σ_{facet i} (b_i − a_i·c) · Vol_{n−1}(proj_k F_i) / |a_{i,k}|

对任意内点 c 成立。按「紧约束集」记忆化，每个面只算一次。

门 / gates: 与 exact_connected.py 的逐项精确分数逐位一致（Φ₄ 已知
−1/60，且其 26 项的分项值已由片枚举法给出）。

Usage:
  python3 exact_polytope.py phi4
  python3 exact_polytope.py cycle B [--terms i:j] [--out F]
  python3 exact_polytope.py 42 D    [--terms i:j] [--out F]
"""
import itertools
import sys
import time
from fractions import Fraction as F

sys.path.insert(0, ".")
from exact_connected import compile_terms

ZERO, ONE = F(0), F(1)


# ---------------------------------------------------------------------
def _solve(M, rhs, n):
    """精确高斯消元；奇异返回 None。"""
    A = [list(M[i]) + [rhs[i]] for i in range(n)]
    for c in range(n):
        p = None
        for r in range(c, n):
            if A[r][c] != 0:
                p = r; break
        if p is None:
            return None
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [v / pv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[i][n] for i in range(n)]


def vertices(rows, n):
    """H-表示 {a·z <= b} 的顶点集（精确有理）。"""
    m = len(rows)
    out = set()
    for comb in itertools.combinations(range(m), n):
        z = _solve([rows[i][0] for i in comb], [rows[i][1] for i in comb], n)
        if z is None:
            continue
        ok = True
        for a, b in rows:
            s = ZERO
            for ai, zi in zip(a, z):
                if ai:
                    s += ai * zi
            if s > b:
                ok = False; break
        if ok:
            out.add(tuple(z))
    return out


def _tight(rows, V):
    """在全部顶点上都取等号的约束下标集合。"""
    t = []
    for i, (a, b) in enumerate(rows):
        if all(sum(ai * zi for ai, zi in zip(a, v) if ai) == b for v in V):
            t.append(i)
    return frozenset(t)


def volume(rows, n, V=None, memo=None):
    """n 维多面体的精确体积；rows 为 H-表示。V 可预先给出顶点。"""
    if memo is None:
        memo = {}
    if V is None:
        V = vertices(rows, n)
    if not V:
        return ZERO
    if n == 0:
        return ONE
    if n == 1:
        xs = [v[0] for v in V]
        return max(xs) - min(xs)
    key = (n, _tight(rows, V), frozenset(V))
    if key in memo:
        return memo[key]
    # 内点：顶点重心
    c = [sum(v[i] for v in V) / len(V) for i in range(n)]
    tot = ZERO
    seen = set()
    for i, (a, b) in enumerate(rows):
        Vi = tuple(sorted(v for v in V
                          if sum(ai * zi for ai, zi in zip(a, v) if ai) == b))
        if len(Vi) < n:                      # 不是 (n−1) 维面
            continue
        if Vi in seen:
            continue
        seen.add(Vi)
        k = next((j for j in range(n) if a[j] != 0), None)
        if k is None:
            continue
        # 把 z_k 代换掉，投影到其余坐标
        ak = a[k]
        pr = []
        for (a2, b2) in rows:
            if a2[k] == 0:
                co = [a2[j] for j in range(n) if j != k]
                if any(co):
                    pr.append((tuple(co), b2))
            else:
                f = a2[k] / ak
                co = [a2[j] - f * a[j] for j in range(n) if j != k]
                rb = b2 - f * b
                if any(co):
                    pr.append((tuple(co), rb))
        Vp = set(tuple(v[j] for j in range(n) if j != k) for v in Vi)
        sub = volume(pr, n - 1, Vp, memo)
        if sub == 0:
            continue
        h = b - sum(ai * ci for ai, ci in zip(a, c) if ai)
        tot += h * sub / abs(ak)
    res = tot / n
    memo[key] = res
    return res


# ---------------------------------------------------------------------
def hrep(A, B, d):
    """t ≤ A_i(x) ≤ t+1 ; s ≤ B_j(x) ≤ s+1 ，变量 (x_0..x_{d-1}, t, s)。"""
    n = d + 2
    rows = []
    for P, idx in ((A, d), (B, d + 1)):
        for p in P:
            c = list(p) + [ZERO, ZERO]
            c[idx] = -ONE
            rows.append((tuple(-v for v in c), ZERO))     # t − A_i ≤ 0
            rows.append((tuple(c), ONE))                  # A_i − t ≤ 1
    return n, rows


def cycle_pos(b, masks):
    d = b - 1
    A = [[ZERO] * d] + [[ONE if j < k else ZERO for j in range(d)]
                        for k in range(1, b)]
    B = [[ZERO] * d]
    for mk in masks:
        co = [ZERO] * d
        for j in mk:
            if j < d:
                co[j] += ONE
            else:
                for i in range(d):
                    co[i] -= ONE
        B.append(co)
    return d, A, B


def spec42_pos(dcls, masks):
    d = 4
    V = [ONE, ZERO, ZERO, ZERO]
    S1 = [ZERO, ONE, ZERO, ZERO]
    S2 = [ZERO, ONE, ONE, ZERO]
    S3 = [ZERO, ONE, ONE, ONE]
    Z = [ZERO] * 4
    ad = lambda a, b: [x + y for x, y in zip(a, b)]
    A = {1: [Z, V, Z, S1, S2, S3],
         2: [Z, V, ad(V, S1), S1, S2, S3],
         3: [Z, V, ad(V, S1), ad(V, S2), S2, S3]}[dcls]
    W = [S1, [x - y for x, y in zip(S2, S1)],
         [x - y for x, y in zip(S3, S2)], [-x for x in S3]]
    B = [Z]
    for mk in masks:
        co = Z[:]
        for j in mk:
            co = ad(co, W[j])
        B.append(co)
    return d, A, B


def weighted_x0(rows, n):
    """∫_P |z_0| dz ：按 z_0 符号切开，各自再提升一维。"""
    tot = ZERO
    for sgn in (ONE, -ONE):
        rs = [(tuple(list(a) + [ZERO]), b) for a, b in rows]
        e0 = [ZERO] * (n + 1); e0[0] = -sgn
        rs.append((tuple(e0), ZERO))                       # sgn·z_0 ≥ 0
        ey = [ZERO] * (n + 1); ey[n] = -ONE
        rs.append((tuple(ey), ZERO))                       # y ≥ 0
        ey2 = [ZERO] * (n + 1); ey2[0] = -sgn; ey2[n] = ONE
        rs.append((tuple(ey2), ZERO))                      # y ≤ sgn·z_0
        tot += volume(rs, n + 1)
    return tot


def run_term(kind, arg, masks):
    if kind == "42":
        d, A, B = spec42_pos(arg, masks)
        n, rows = hrep(A, B, d)
        return weighted_x0(rows, n)
    d, A, B = cycle_pos(arg, masks)
    n, rows = hrep(A, B, d)
    return volume(rows, n)


if __name__ == "__main__":
    what = sys.argv[1]
    pos = [a for a in sys.argv[2:] if not a.startswith("--")]
    lo, hi, out = 0, 10 ** 9, None
    if "--terms" in sys.argv:
        lo, hi = (int(x) for x in
                  sys.argv[sys.argv.index("--terms") + 1].split(":"))
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    if what == "phi4":
        kind, arg, terms, tgt = "cycle", 4, compile_terms(4), F(-1, 60)
    elif what == "cycle":
        arg = int(pos[0]); kind = "cycle"; terms = compile_terms(arg)
        tgt = {4: F(-1, 60), 5: F(1, 36), 6: F(-1, 126)}.get(arg)
    elif what == "42":
        arg = int(pos[0]); kind = "42"; terms = compile_terms(4)
        tgt = {1: F(-2, 315), 2: F(-1, 1260), 3: F(-1, 252)}[arg]
    else:
        print(__doc__); sys.exit(1)
    hi = min(hi, len(terms))
    fh = open(out, "a") if out else None
    tot = ZERO
    t0 = time.time()
    for i in range(lo, hi):
        sign, masks = terms[i]
        ts = time.time()
        v = run_term(kind, arg, masks)
        tot += sign * v
        line = (f"poly {what}{pos} term {i} sign {sign} value {v} "
                f"wall {time.time()-ts:.2f}")
        print(line, flush=True)
        if fh:
            fh.write(line + "\n"); fh.flush()
    if fh:
        fh.close()
    print(f"# total [{lo},{hi}) = {tot} = {float(tot):.15f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    if tgt is not None and lo == 0 and hi == len(terms):
        print(f"# target {tgt}  {'MATCH' if tot == tgt else 'MISMATCH'}")
