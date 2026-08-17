import numpy as np, math
rng = np.random.default_rng(7)
mks = []
for rep in range(20):
    Ng = 1000
    H = rng.normal(size=(Ng,Ng)) + 1j*rng.normal(size=(Ng,Ng))
    H = (H + H.conj().T)/2
    ev = np.linalg.eigvalsh(H)
    # unfold by semicircle CDF: entries var 1 => radius 2*sqrt(Ng)
    R = 2*math.sqrt(Ng); x = np.clip(ev/R, -1, 1)
    F = Ng*(0.5 + (x*np.sqrt(1-x**2) + np.arcsin(x))/math.pi)
    u = np.sort(F)
    mid = u[(u > Ng*0.5 - 150) & (u < Ng*0.5 + 150)]   # central ~300 unfolded points
    d = len(mid)
    D = mid[:,None] - mid[None,:]
    A = np.sinc(D)          # sin(pi x)/(pi x), diag = 1
    m = [np.trace(np.linalg.matrix_power(A,k))/d for k in [1,2,3,4]]
    mks.append(m)
mks = np.array(mks)
mean = mks.mean(0); se = mks.std(0)/math.sqrt(len(mks))
print("GUE sine-process simulation (20 reps, ~300 pts each):")
for k,(mu,s) in enumerate(zip(mean,se),1):
    tgt = [1.0, 4/3, 2.0, 13/4][k-1]
    print(f"  m{k} = {mu:.4f} +- {s:.4f}   target {tgt:.4f}   dev {(mu-tgt)/tgt*100:+.2f}%")
