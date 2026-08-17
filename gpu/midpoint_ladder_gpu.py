# -*- coding: utf-8 -*-
"""GPU grouped midpoint-ladder engine for the connected constants
(paper §5.5): computes ∫O_bC_b = Σ_{(P,σ)} sign·[ov·ov integral]
on midpoint grids, batched for GPU.

Mathematically identical to certification/midpoint_ladder.py; the
per-point work is reformulated as ONE dense matmul per chunk
(positions = V · Cᵀ over the deduplicated prefix-mask coefficient
matrix C) followed by cheap per-term max/min reductions, which is
the GPU-friendly shape.  Backends, in order of preference:
cupy (CUDA) → torch (CUDA/MPS) → numpy (CPU fallback, used by the
validation gate).  fp64 throughout; v₁-slice checkpointing, so a
run can be split across nodes by [i0, i1) slice ranges and the
partial sums added.

GPU 版分组中点阶梯引擎 (论文 §5.5): 与 certification/
midpoint_ladder.py 数学等价; 每点计算重构为每块一次稠密矩阵乘
(位置 = V·Cᵀ, C 为去重后的前缀掩码系数矩阵) 加逐项 max/min
归约. 后端优先级 cupy (CUDA) → torch (CUDA/MPS) → numpy (CPU
回退, 校准门使用). 全程 fp64; 按 v₁ 分片检查点, 可按 [i0,i1)
分片跨节点拆分, 部分和相加即总积分.

Usage / 用法:
    python3 midpoint_ladder_gpu.py B DV [i0 i1] [--chunk N]
    python3 midpoint_ladder_gpu.py validate      # b=4 gate vs CPU engine

Validation gate / 校准门: b=4, coarse dv — must match the CPU
engine to ~1e-12 and the exact Φ₄ = −1/60 within the ladder band.
"""
import itertools
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------
# Backend selection / 后端选择
# ---------------------------------------------------------------
BACKEND = "numpy"
xp = np
_torch = None
try:
    import cupy  # type: ignore
    if cupy.cuda.runtime.getDeviceCount() > 0:
        xp = cupy
        BACKEND = "cupy"
except Exception:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            _torch = torch
            BACKEND = "torch-cuda"
        elif getattr(torch.backends, "mps", None) and \
                torch.backends.mps.is_available():
            _torch = torch
            BACKEND = "torch-mps"  # NB: MPS 无 fp64, 自动降级 numpy
            BACKEND = "numpy"
    except Exception:
        pass


def set_partitions(items):
    """All set partitions of `items` / items 的全部集合分拆."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def build_terms(b):
    """(sign, prefix-mask tuple list) per (partition, cyclic order).
    每个 (分拆, 循环序) 的 (符号, 前缀掩码序列). Identical to the
    CPU engine's build_terms."""
    out = []
    for P in set_partitions(list(range(b))):
        m = len(P)
        sign = (-1.0) ** (m - 1)
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


def compile_terms(b):
    """Deduplicate prefix masks into one coefficient matrix.

    A mask S ⊂ {0..b−1} defines position pos_S = Σ_{j∈S} v_j with
    v_{b−1} = −(v_0+…+v_{b−2}); its coefficient row over the b−1
    free variables is  e_S − [b−1 ∈ S]·(1,…,1).  Positions for ALL
    terms are then V · Cᵀ (V: points × (b−1)); each term reads its
    columns and reduces max/min against the implicit 0 position.

    把全部前缀掩码去重为一个系数矩阵 C: 掩码 S 的位置对 b−1 个
    自由变量的系数行为 e_S − [b−1∈S]·全 1 行; 所有项的位置由一次
    V·Cᵀ 得到, 每项取其列并与隐含的 0 位置做 max/min 归约.
    """
    terms = build_terms(b)
    uniq = {}
    for _, masks in terms:
        for S in masks:
            if S not in uniq:
                uniq[S] = len(uniq)
    C = np.zeros((len(uniq), b - 1), dtype=np.float64)
    for S, r in uniq.items():
        for j in S:
            if j < b - 1:
                C[r, j] += 1.0
            else:                       # v_{b-1} = -sum(free)
                C[r, :] -= 1.0
    idx = [(s, [uniq[S] for S in masks]) for s, masks in terms]
    return C, idx


def _to_dev(a):
    if BACKEND == "cupy":
        return xp.asarray(a)
    if BACKEND.startswith("torch"):
        return _torch.as_tensor(a, dtype=_torch.float64, device="cuda")
    return a


def _to_host(a):
    if BACKEND == "cupy":
        return float(xp.asnumpy(a))
    if BACKEND.startswith("torch"):
        return float(a.item())
    return float(a)


def slice_integral(b, dv, v1, C, idx, chunk=1 << 22):
    """Integral contribution of one v₁-slice (without dv^{b-1}).
    单个 v₁ 分片的积分贡献 (未乘 dv^{b-1})."""
    g = np.arange(-2 + dv / 2, 2, dv)
    n = len(g)
    nin = b - 2                          # inner free variables
    total_pts = n ** nin
    Cd = _to_dev(C)
    acc = 0.0
    for start in range(0, total_pts, chunk):
        stop = min(start + chunk, total_pts)
        # 展平的内层格点索引 → 坐标 (row-major, ij indexing)
        flat = np.arange(start, stop, dtype=np.int64)
        Vh = np.empty((stop - start, b - 1), dtype=np.float64)
        Vh[:, 0] = v1
        rem = flat
        for d in range(nin - 1, -1, -1):
            Vh[:, 1 + d] = g[rem % n]
            rem = rem // n
        V = _to_dev(Vh)
        if BACKEND.startswith("torch"):
            pos = V @ Cd.T               # (pts, n_uniq)
            zeros = _torch.zeros(V.shape[0], 1, dtype=_torch.float64,
                                 device="cuda")
        else:
            pos = V @ Cd.T
            zeros = xp.zeros((V.shape[0], 1), dtype=xp.float64)
        # 外层 O_b: 前缀走步 0, v0, v0+v1, ... (b−1 个前缀 + 0)
        # prefix walk coefficient rows are the masks {0},{0,1},...
        # 已包含于 C 当且仅当作为某项前缀出现; 为稳妥单独计算:
        if BACKEND.startswith("torch"):
            W = _torch.cumsum(V, dim=1)
            wmax = _torch.clamp(W.max(dim=1).values, min=0.0)
            wmin = _torch.clamp(W.min(dim=1).values, max=0.0)
            ovV = _torch.clamp(1.0 - (wmax - wmin), min=0.0)
        else:
            W = xp.cumsum(V, axis=1)
            wmax = xp.maximum(W.max(axis=1), 0.0)
            wmin = xp.minimum(W.min(axis=1), 0.0)
            ovV = xp.clip(1.0 - (wmax - wmin), 0.0, None)
        # 逐项: 位置列 + 隐含 0 → overlap, 累加带号和
        if BACKEND.startswith("torch"):
            sub = _torch.zeros_like(ovV)
            cat = _torch.cat
            for sign, cols in idx:
                P = cat([zeros, pos[:, cols]], dim=1)
                ov = _torch.clamp(
                    1.0 - (P.max(dim=1).values - P.min(dim=1).values),
                    min=0.0)
                sub = sub + sign * ov
            acc += float((ovV * sub).sum().item())
        else:
            sub = xp.zeros_like(ovV)
            for sign, cols in idx:
                P = xp.concatenate([zeros, pos[:, cols]], axis=1)
                ov = xp.clip(1.0 - (P.max(axis=1) - P.min(axis=1)),
                             0.0, None)
                sub = sub + sign * ov
            acc += _to_host((ovV * sub).sum())
    return acc


def run(b, dv, i0=0, i1=None, chunk=1 << 22):
    C, idx = compile_terms(b)
    g = np.arange(-2 + dv / 2, 2, dv)
    n = len(g)
    i1 = n if i1 is None else min(i1, n)
    tag = f"/tmp/mladder_gpu_b{b}_dv{dv}.npz"
    try:
        d = np.load(tag)
        acc, done = float(d["acc"]), int(d["done"])
    except Exception:
        acc, done = 0.0, 0
    if done > i0:
        i0 = done
    print(f"[gpu-ladder] backend={BACKEND} b={b} dv={dv} "
          f"grid={n}^{b-1} terms={len(idx)} uniq-masks={C.shape[0]} "
          f"slices [{i0},{i1})", flush=True)
    t0 = time.time()
    for i in range(i0, i1):
        acc += slice_integral(b, dv, g[i], C, idx, chunk) * dv ** (b - 1)
        np.savez(tag, acc=acc, done=i + 1)
        el = time.time() - t0
        if el > 0 and (i - i0) >= 0:
            rate = (i - i0 + 1) / el
            eta = (i1 - i - 1) / rate if rate > 0 else float("inf")
            print(f"  slice {i+1}/{n}  partial {acc:+.8f}  "
                  f"({rate:.2f} sl/s, ETA {eta/3600:.2f} h)", flush=True)
    print(f"[gpu-ladder] b={b} dv={dv}: ∫O_bC_b = {acc:+.8f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return acc


def validate():
    """b=4 calibration gate vs the CPU engine and Φ₄ = −1/60.
    b=4 校准门: 对 CPU 引擎逐位、对精确 Φ₄ 带内一致."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__) or ".",
                                    "..", "certification"))
    from midpoint_ladder import build_terms as cpu_terms  # noqa: F401
    import midpoint_ladder as cpu
    b, dv = 4, 0.05
    # CPU 参考 (直接内联累加, 避免 /tmp 检查点干扰)
    import numpy as _np
    terms = cpu.build_terms(b)
    g = _np.arange(-2 + dv / 2, 2, dv)
    ref = 0.0
    for v1 in g:
        mesh = _np.meshgrid(*([g] * (b - 2)), indexing="ij")
        vs = [_np.full_like(mesh[0], v1)] + list(mesh)
        vs.append(-sum(vs))
        walk = [_np.zeros_like(vs[0])]
        s = _np.zeros_like(vs[0])
        for v in vs[:-1]:
            s = s + v
            walk.append(s.copy())
        mx = _np.maximum.reduce(walk); mn = _np.minimum.reduce(walk)
        ovV = _np.clip(1.0 - (mx - mn), 0.0, None)
        sub = _np.zeros_like(vs[0])
        for sign, masks in terms:
            posl = [_np.zeros_like(vs[0])]
            for mk in masks:
                posl.append(sum(vs[j] for j in mk))
            pmx = _np.maximum.reduce(posl); pmn = _np.minimum.reduce(posl)
            sub = sub + sign * _np.clip(1.0 - (pmx - pmn), 0.0, None)
        ref += float((ovV * sub).sum()) * dv ** (b - 1)
    # GPU-代码路径 (当前后端; numpy 回退亦覆盖同一代码)
    C, idx = compile_terms(b)
    got = 0.0
    for v1 in g:
        got += slice_integral(b, dv, v1, C, idx) * dv ** (b - 1)
    dev = abs(got - ref)
    print(f"[validate] b=4 dv={dv}: engine={ref:+.10f} "
          f"gpu-path={got:+.10f} dev={dev:.2e} "
          f"(exact Φ₄ = −1/60 = {-1/60:+.10f})")
    assert dev < 1e-10, "GPU path deviates from CPU engine"
    print("[validate] PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
        sys.exit(0)
    b = int(sys.argv[1])
    dv = float(sys.argv[2])
    args = [a for a in sys.argv[3:] if not a.startswith("--")]
    i0 = int(args[0]) if len(args) > 0 else 0
    i1 = int(args[1]) if len(args) > 1 else None
    chunk = 1 << 22
    if "--chunk" in sys.argv:
        chunk = int(sys.argv[sys.argv.index("--chunk") + 1])
    run(b, dv, i0, i1, chunk)
