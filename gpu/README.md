# GPU module — the cluster-scale N1 computation
# GPU 模块 —— 集群级 N1 计算

Target (paper §5.5, §11 item N1): tighten and ultimately identify
the three connected constants C₅, {4,2}, {6}. They are **proved
rational** (dimension theorem, paper §5.5), so the endgame is
fraction identification, not real-number enclosure. The strategy
here is two-stage:
目标（论文 §5.5 与 §11 之 N1 项）：收紧并最终识别三个连通常数
C₅、{4,2}、{6}。三者**已证为有理数**（维数定理），因此终局是
识别分数而非包住实数。两段式策略：

1. **GPU ladders** (`midpoint_ladder_gpu.py`) — the grouped
   midpoint-grid ladders at much finer steps than CPU allows.
   Mathematically identical to `certification/midpoint_ladder.py`
   (validation gate: byte-identical at b=4); reformulated as one
   dense fp64 matmul per chunk + per-term max/min reductions.
   GPU 阶梯：与 CPU 引擎数学等价（b=4 校准门零偏差），重构为
   每块一次 fp64 矩阵乘加逐项归约。
2. **Rational reconstruction** (`rational_reconstruct.py`) — feed
   the converged value + band into continued-fraction search; a
   unique small-height candidate becomes the target for the exact
   integrator (`certification/exact_t222.py` methodology) to
   verify, collapsing the cluster-scale exact run into a single
   verification. 有理重构：收敛值+带宽 → 连分数搜索 → 唯一小
   高度候选交精确积分器验证（把集群级盲算收缩为单次验证）。
   Gate check: `python3 rational_reconstruct.py -0.0166667 2e-5`
   returns −1/60; `0.3119048 1e-6` returns 131/420.

## Run / 运行

```bash
python3 midpoint_ladder_gpu.py validate      # b=4 gate, must PASS
python3 midpoint_ladder_gpu.py 5 0.016       # C5 ladder rung
python3 midpoint_ladder_gpu.py 6 0.02        # {6} ladder rung
# multi-node split by v1-slices: node k of K runs [k*n/K,(k+1)*n/K)
python3 midpoint_ladder_gpu.py 6 0.01 0 100  # slices [0,100)
```

Backend auto-selects cupy (CUDA) → torch (CUDA) → numpy.
Checkpoints in /tmp/mladder_gpu_*.npz; partial sums from disjoint
slice ranges add up to the total (embarrassingly parallel).
后端自动选择；检查点续算；不相交分片的部分和相加即总积分，
天然多机并行。

## Compute estimates / 算力规模估计

Cost model: grid |g| = 4/dv midpoints per dimension; points =
|g|^(b−1); ~(n_uniq·(b−1) + Σ_terms m) fp64 flops/point ≈
1.5e3 (b=5, 150 terms), 6e3 (b=6, 1082 terms), 5e4 (b=7, 9366
terms). A100 80GB sustained fp64 (with matmul on tensor cores)
taken conservatively at 2–5 TFLOP/s effective.
成本模型如上；A100 有效 fp64 按 2–5 TFLOP/s 保守估计。

| run 目标 | dv | points | FLOPs | 单卡 A100 | 8×A100 |
|---|---|---|---|---|---|
| C₅ (b=5) | 0.016 (\|g\|=250) | 3.9e9 | ~6e12 | 分钟级 | — |
| C₅ (b=5) | 0.004 (\|g\|=1000) | 1.0e12 | ~1.5e15 | ~10–30 min | 数分钟 |
| {6} (b=6) | 0.02 (\|g\|=200) | 3.2e11 | ~2e15 | ~15–30 min | 数分钟 |
| {6} (b=6) | 0.01 (\|g\|=400) | 1.0e13 | ~6e16 | ~4–8 h | ~0.5–1 h |
| {6} (b=6) | 0.005 (\|g\|=800) | 3.3e14 | ~2e18 | ~5–10 天 | ~15–30 h |
| C₇ (b=7, 侦察) | 0.05 (\|g\|=80) | 2.6e11 | ~1.3e16 | ~1–2 h | ~10–20 min |
| C₇ (b=7) | 0.025 (\|g\|=160) | 1.7e13 | ~8e17 | ~2–5 天 | ~6–15 h |

Practical recommendation / 实用建议:
- **First target 首选目标**: {6} at dv = 0.02/0.01/0.005 — a
  three-rung Richardson ladder shrinking the ±0.0008 band by
  ~an order of magnitude; feeds directly into the LP corner
  (paper A5 sensitivity ∂/∂(u₄₂+u₆) = −0.93) and gives
  `rational_reconstruct` a real shot at the fraction.
  {6} 三档阶梯：把 ±0.0008 带宽压一个量级，直接改善 LP 角点，
  并给有理重构以实际机会。
- C₅ at dv=0.004 is cheap insurance tightening ±0.0001 further.
  C₅ 加密为廉价保险。
- {4,2} needs the spectator-pair variant of the engine (same
  matmul pattern, product of two overlap factors); port on
  request once the b-cycle runs are confirmed on your hardware.
  {4,2} 需旁观对变体（同一矩阵乘模式）；待 b-圈运行在您的硬件
  确认后可随即移植。
- Memory: chunked (default 2^22 points/chunk ≈ 1.3 GB fp64
  workspace at b=6); any 16 GB+ GPU works, chunk size adjustable
  via `--chunk`. 显存分块可调，16 GB 以上任意卡可跑。
- fp64 accumulation error at 1e13 points ~ 1e-3·ulp scale —
  negligible против the dv-band; slice partial sums are
  checkpointed in float64 and added exactly once.

Deliverable back to the program / 回传格式: for each run report
(b, dv, slices, final ∫O_bC_b to 8 decimals, wall time, GPU
model). Three rungs per constant suffice for the Richardson
extrapolation and the reconstruction attempt.
每次运行回报 (b, dv, 分片数, 8 位小数的积分值, 墙钟, GPU 型号)；
每常数三档即可做 Richardson 外推与重构尝试。
