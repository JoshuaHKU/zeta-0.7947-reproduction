import numpy as np, math, time
t0 = time.time()
rng = np.random.default_rng(11)
def sample_moments(Wpts, reps, Ng):
    out = []
    for rep in range(reps):
        H = rng.normal(size=(Ng,Ng)) + 1j*rng.normal(size=(Ng,Ng))
        H = (H + H.conj().T)/2
        ev = np.linalg.eigvalsh(H)
        R = 2*math.sqrt(Ng); x = np.clip(ev/R, -1, 1)
        F = Ng*(0.5 + (x*np.sqrt(1-x**2) + np.arcsin(x))/math.pi)
        u = np.sort(F)
        mid = u[(u > Ng*0.5 - Wpts/2) & (u < Ng*0.5 + Wpts/2)]
        d = len(mid)
        A = np.sinc(mid[:,None] - mid[None,:])
        ee = np.linalg.eigvalsh(A)
        out.append([np.sum(ee**k)/d for k in range(1,7)])
    return np.array(out)

M200 = sample_moments(200, 60, 900)
M400 = sample_moments(400, 60, 1500)
m200, s200 = M200.mean(0), M200.std(0)/math.sqrt(60)
m400, s400 = M400.mean(0), M400.std(0)/math.sqrt(60)
mext = 2*m400 - m200
sext = np.sqrt(4*s400**2 + s200**2)
tgt = [1, 4/3, 2, 13/4, None, None]
print("k   m(W=200)    m(W=400)    Richardson     se       target")
for k in range(6):
    t = f"{tgt[k]:.4f}" if tgt[k] else "  new"
    print(f"{k+1}   {m200[k]:8.4f}   {m400[k]:8.4f}   {mext[k]:8.4f}   {sext[k]:.4f}   {t}")
print(f"({time.time()-t0:.0f}s)")
