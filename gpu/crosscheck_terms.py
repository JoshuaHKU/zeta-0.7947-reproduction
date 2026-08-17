# -*- coding: utf-8 -*-
"""逐项数值粗对照：exact_connected.py 的每一项 vs 参考实现的 numpy 直算。
Per-term numeric cross-check of exact_connected.py against a numpy
evaluation built from spectator_42_reference.py's own primitives.

目的不是精度，而是**尽早暴露符号/掩码错误**：如果 Cls 的位置族或符号
构造与规格不符，粗网格上的数值就会在第 2–3 位小数上对不上，而不必等
到某一项的精确积分跑完几小时之后。

Usage:
  python3 crosscheck_terms.py 42 D DV     # {4,2} 第 D 放置类
  python3 crosscheck_terms.py cycle B DV  # 纯 b-圈
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from exact_connected import spec42_terms, cycle_terms, aff_eval


def numeric_term(cls, dv):
    """在中点网格上直接数值积分单个 Cls（与精确积分器同一被积函数定义，
    但走完全不同的代码路径：numpy 全网格 + 显式 max/min）。"""
    g = np.arange(-1 + dv / 2, 1, dv)          # 支撑含于 [-1,1]^d
    d = cls.d
    mesh = np.meshgrid(*([g] * d), indexing="ij")

    def ev(p):
        r = np.full_like(mesh[0], float(p[0]))
        for i in range(d):
            c = float(p[i + 1])
            if c:
                r = r + c * mesh[i]
        return r

    A = [ev(p) for p in cls.posA]
    sa = np.maximum.reduce(A) - np.minimum.reduce(A)
    B = [ev(p) for p in cls.posB]
    sb = np.maximum.reduce(B) - np.minimum.reduce(B)
    val = np.clip(1 - sa, 0, None) * np.clip(1 - sb, 0, None)
    if cls.wsign is not None:
        val = val * np.minimum(np.abs(mesh[cls.wsign]), 1.0)
    return float(val.sum()) * dv ** d


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "42":
        D = int(sys.argv[2]); terms = spec42_terms(D); lab = f"42 d={D}"
    else:
        b = int(sys.argv[2]); terms = cycle_terms(b); lab = f"cycle b={b}"
    dv = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    print(f"# {lab}, {len(terms)} terms, numeric midpoint dv={dv}")
    tot = 0.0
    for i, (sign, cls) in enumerate(terms):
        v = numeric_term(cls, dv)
        tot += sign * v
        print(f"term {i:4d} sign {sign:+d} numeric {v:+.9f}")
    print(f"# numeric class total = {tot:+.9f}")
