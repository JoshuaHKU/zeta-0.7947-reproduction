# -*- coding: utf-8 -*-
"""Rational reconstruction of the connected constants from
high-precision ladder values (paper §5.5 dimension theorem: C5,
{4,2}, {6} are PROVED rational, so certification means identifying
a fraction).  Given a converged ladder value x with error bound
eps, enumerate continued-fraction convergents and nearby small-
denominator fractions p/q with |x - p/q| <= eps; a unique
candidate at small height is the reconstruction target for the
exact integrator to verify.

由高精度阶梯值重构连通常数的有理数候选 (论文 §5.5 维数定理已证
C5, {4,2}, {6} 为有理数, 故认证即识别分数): 给定收敛阶梯值 x 与
误差界 eps, 枚举连分数渐近分数及邻近的小分母分数 p/q 满足
|x - p/q| <= eps; 小高度下的唯一候选即精确积分器的验证目标.

Usage / 用法:
    python3 rational_reconstruct.py X EPS [QMAX]
e.g. 校准示例 (b=4 精确值 -1/60):
    python3 rational_reconstruct.py -0.016667 2e-5
"""
import sys
from fractions import Fraction


def convergents(x, qmax):
    """Continued-fraction convergents of x with denominator <= qmax.
    x 的连分数渐近分数 (分母 <= qmax)."""
    out = []
    a = x
    p0, q0, p1, q1 = 0, 1, 1, 0
    for _ in range(64):
        ai = int(a // 1)
        p0, q0, p1, q1 = p1, q1, ai * p1 + p0, ai * q1 + q0
        if q1 > qmax:
            break
        out.append(Fraction(p1, q1))
        frac = a - ai
        if abs(frac) < 1e-15:
            break
        a = 1.0 / frac
    return out


def candidates(x, eps, qmax=10 ** 6):
    """All fractions p/q, q <= qmax, with |x - p/q| <= eps, collected
    from convergents of x, x±eps (deduplicated, sorted by q).
    收集 x, x±eps 的渐近分数中满足误差界的候选, 按分母排序去重."""
    seen = {}
    for base in (x, x - eps, x + eps):
        for f in convergents(base, qmax):
            if abs(x - float(f)) <= eps and f not in seen:
                seen[f] = abs(x - float(f))
    return sorted(seen.items(), key=lambda kv: (kv[0].denominator,
                                                abs(kv[0].numerator)))


if __name__ == "__main__":
    x = float(sys.argv[1])
    eps = float(sys.argv[2])
    qmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 6
    cs = candidates(x, eps, qmax)
    print(f"x = {x}  eps = {eps}  qmax = {qmax}")
    if not cs:
        print("no candidate within eps at this qmax — tighten the "
              "ladder or raise qmax / 无候选: 加密阶梯或提高 qmax")
    for f, d in cs[:12]:
        print(f"  {str(f):>20s}  = {float(f):+.12f}   |dev| = {d:.2e}")
    if len(cs) == 1:
        print("UNIQUE candidate at this precision — hand to the exact "
              "integrator for verification / 唯一候选, 交精确积分器验证")
