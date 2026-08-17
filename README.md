# More than 0.7947 of the zeros of the Riemann zeta function are simple and on the critical line

Preprint + self-contained reproduction, certification and Lean
packages. / 预印本与自包含的复现、认证及 Lean 形式化包。

Repository / 仓库: https://github.com/JoshuaHKU/zeta-0.7947-reproduction

Built on the two-thirds preprint
[[C26]](https://www-cdn.anthropic.com/564f962e60643842f5fcb4a17c9dbc8f608f1c37.pdf)
(Claude, Anthropic, August 2026); all other references are published
literature. Authors: Hongyi Yang, Shihua Yang; mathematical
development by Claude (Anthropic) — see the paper's Acknowledgements.
本文以公开预印本 [C26]（两三定理）为基础；其余引用均为已发表文献。

## Layout / 目录结构

| Path | Contents / 内容 |
|---|---|
| `paper.tex` / `paper.pdf` | The paper (33 pp.) / 论文全文 |
| `REPRODUCTION.md` | Complete reproduction checklist with recorded outputs / 完整复现清单与记录输出 |
| `certification/` | Exact-rational certification: dual certificate (`certify_lp.py`), exact pairing integrals (`exact_t222.py`), certificate family (`certificate_family.py`), convolution-calculus gates (`cyclic_cumulant.py`), midpoint ladders (`midpoint_ladder.py`) / 精确有理认证层 |
| `pipeline/` | Identity & face gate suite, `python3 run_all.py` (~60 s, ALL PASS) / 恒等式与数值面门套件 |
| `scripts/` | Constant suites and calibration faces (§3–§4 gates, GUE bands, sawtooth, quadrature) / 常数套件与校准面 |
| `lean/RhGate/` | Core Lean 4 package: 4 modules, 20 theorems, 0 `sorry`, no mathlib; identity, local-law and certificate layers kernel-checked / 核心 Lean 4 包 |
| `gpu/` | GPU midpoint-ladder engine for the cluster-scale N1 computation (cupy/torch/numpy backends, validated against the CPU engine) + rational reconstruction / N1 集群级计算的 GPU 引擎与有理重构 |
| `paper-zh.tex` / `paper-zh.pdf` | Chinese translation (XeLaTeX; the English version is authoritative) / 中文译本（以英文版为准） |

## Quick start / 快速开始

```bash
# exact dual certificate (< 1 s) / 精确对偶证书
cd certification && python3 certify_lp.py

# full gate suite (~60 s) / 全部数值门
cd pipeline && python3 run_all.py

# Lean kernel check / Lean 内核验证
cd lean/RhGate && lake build   # toolchain: leanprover/lean4:v4.33.0
```

Headline, certified in exact rational arithmetic and kernel-checked
in core Lean 4 (the rational-arithmetic layer):
**N₀ˢ/N ≥ 1 − 2w₀ ≥ 0.7947, N_d/N ≥ 1 − w₀ ≥ 0.8973**, with
w₀ = 1153107070889/11233957316589.

## Claim grade / 声明等级

Certified-candidate, not record (paper §11): the three connected
constants carry stated numerical bands (each proved rational, closed
forms open); the analytic layer is unformalized; external review
pending. Verification registry: 194 pre-registered checks, 34 fired
and converted, all forward-recorded (paper Appendix).
声明等级为 certified-candidate（论文 §11）：三个连通常数带数值带宽
（已证为有理数，闭形式待求）；解析层未形式化；外部评审待进行。
