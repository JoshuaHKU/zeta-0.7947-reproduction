# -*- coding: utf-8 -*-
"""m4 assembly with product-scale band split (paper §3 model gates):
the connected k=4 integral evaluated bandwise; total 3.25103 vs 13/4.
m4 装配与积尺度分带 (论文 §3 模型门): k=4 连通积分分带求值;
总量 3.25103 对照 13/4.
"""
import numpy as np, itertools, math, time
# reuse the Ursell engine definitions (overlap, C_of) from the shared
# module / 复用共享模块的 Ursell 引擎定义
exec(open('cumulant_engine.py').read().split("# gate b=2")[0])
t0=time.time()

# NOTE: the pairing anchors printed below are the historical
# endpoint-grid measurements; the exact values are t_adj = 7/60,
# t_opp = 1/30 (paper D7).  The assembly total 3.25103 is the §3
# model-gate figure quoted in the paper.
# 注: 下方配对锚为历史端点网格测量值; 精确值 t_adj = 7/60,
# t_opp = 1/30 (论文 D7). 装配总量 3.25103 即论文 §3 模型门数字.

# ---- k=4 connected integral with product-scale band split ----
dv = 0.02
g = np.arange(-2, 2+dv/2, dv); n1 = len(g)
Iconn_tot = 0.0; bands = {'(0,1]':0.0, '(1,1.2]':0.0, '(1.2,2]':0.0}
prof_s = np.zeros(101)  # s-profile histogram, ds=0.02
for i, v1 in enumerate(g):
    V2, V3 = np.meshgrid(g, g, indexing='ij')
    V1 = np.full_like(V2, v1); V4 = -V1-V2-V3
    O4 = overlap([np.zeros_like(V1), V1, V1+V2, V1+V2+V3])
    m = O4 > 0
    if not m.any(): continue
    C4 = C_of([V1[m], V2[m], V3[m], V4[m]])
    w = O4[m]*C4*dv**3
    s = 0.5*(np.abs(V1[m])+np.abs(V2[m])+np.abs(V3[m])+np.abs(V4[m]))
    Iconn_tot += w.sum()
    bands['(0,1]'] += w[(s > 1e-9) & (s <= 1.0)].sum()
    bands['(1,1.2]'] += w[(s > 1.0) & (s <= 1.2)].sum()
    bands['(1.2,2]'] += w[s > 1.2].sum()
    idx = np.clip((s/0.02).astype(int), 0, 100)
    np.add.at(prof_s, idx, w)
print(f"I_conn total = {Iconn_tot:+.5f}   ({time.time()-t0:.0f}s)")
print("band masses:", {k: f"{x:+.5f}" for k,x in bands.items()})
print("  ==> B_true (hard band s in (1,6/5]) =", f"{bands['(1,1.2]']:+.5f}")

# ---- {2,2} pairing terms ----
du=0.002; u=np.arange(-1,1+du/2,du)
C2u = np.minimum(np.abs(u),1.0)
tt = np.arange(-0.5,0.5+du/2,du)
gfun = np.array([np.sum(C2u[(u>=-0.5-t)&(u<=0.5-t)]*du) for t in tt])
t_adj = np.sum(gfun**2)*du
# opposite: free (theta4, v1, v2), theta-partials {0,v1,v1+v2,v2}
dv2=0.004; vv=np.arange(-1,1+dv2/2,dv2)
Va,Vb = np.meshgrid(vv,vv,indexing='ij')
Oop = overlap([np.zeros_like(Va), Va, Va+Vb, Vb])
t_opp = np.sum(Oop*np.minimum(np.abs(Va),1)*np.minimum(np.abs(Vb),1))*dv2*dv2
print(f"\npairing terms: adjacent = {t_adj:.5f} (x2), opposite = {t_opp:.5f}")
total = 1.0 + 2.0 + 2*t_adj + t_opp + 0.0 + Iconn_tot
print(f"assembly: 1(means) + 2(singleton-pairs) + {2*t_adj+t_opp:.5f}(2+2) + 0(3+1) "
      f"+ {Iconn_tot:+.5f}(conn) = {total:.5f}   [target 13/4 = 3.25]")
