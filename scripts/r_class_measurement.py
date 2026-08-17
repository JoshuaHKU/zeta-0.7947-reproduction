import numpy as np, math, time, sys

def primepowers(X):
    lam = {}
    for p in range(2, X+1):
        if all(p % q for q in range(2, int(p**0.5)+1)):
            pk = p
            while pk <= X:
                lam[pk] = math.log(p); pk *= p
    ns = np.array(sorted(lam), dtype=np.int64); ls = np.array([lam[n] for n in ns])
    return ns, ls

def O4(L, A1, A2, A3):
    z = np.zeros_like(A1)
    mx = np.maximum.reduce([z, A1, A2, A3]); mn = np.minimum.reduce([z, A1, A2, A3])
    return np.clip(L - (mx - mn), 0, None)

def measure(T, C=40.0):
    L = math.log(T/(2*math.pi)); X = int(T/(2*math.pi)); ell1 = L + 2*math.log(2) - 1
    d = L*T/(2*math.pi); norm4 = d*ell1**4; norm2 = d*ell1**2
    ns, ls = primepowers(X); m = len(ns)
    u = np.log(ns.astype(float)); w = ls/np.sqrt(ns.astype(float))
    gate = (T/math.pi)*np.sum(ls**2/ns.astype(float)*np.clip(L-u,0,None))/norm2
    S = (u[:,None]+u[None,:]).ravel(); W2 = (w[:,None]*w[None,:]).ravel()
    U1 = np.repeat(u,m); U2 = np.tile(u,m)
    NN = (ns[:,None]*ns[None,:]).ravel()
    o = np.argsort(S, kind='stable')
    Ss,W2s,U1s,U2s,Ns = S[o],W2[o],U1[o],U2[o],NN[o]
    win = C/T
    lo = np.searchsorted(Ss, Ss-win, 'left'); hi = np.searchsorted(Ss, Ss+win, 'right')
    cnt = hi-lo
    res = {'exact_pair':0.0,'exact_nonpair':0.0,'hne0':0.0}
    B = 200000
    row = 0
    while row < len(Ss):
        r1 = row
        # take rows until match-count budget reached
        c = 0; r2 = r1
        while r2 < len(Ss) and c + cnt[r2] <= B:
            c += cnt[r2]; r2 += 1
        if r2 == r1: r2 = r1+1; c = cnt[r1]
        reps = cnt[r1:r2]
        i_idx = np.repeat(np.arange(r1,r2), reps)
        # vectorized ragged ranges:
        offs = np.concatenate([[0], np.cumsum(reps)])
        j_idx = lo[r1:r2].repeat(reps) + (np.arange(offs[-1]) - offs[:-1].repeat(reps))
        row = r2
        Delta = Ss[i_idx]-Ss[j_idx]
        Wq = W2s[i_idx]*W2s[j_idx]
        u1,u3 = U1s[i_idx],U2s[i_idx]; u2,u4 = U1s[j_idx],U2s[j_idx]
        Nl,Nr = Ns[i_idx],Ns[j_idx]
        sd = np.where(np.abs(Delta)<1e-15, 1.0, Delta)
        Wwin = np.where(np.abs(Delta)<1e-15, T, (np.sin(2*T*Delta)-np.sin(T*Delta))/sd)
        Osum = O4(L,-u1+Delta,u3-u4,-u4) + O4(L,u2-u3-u4,-u3-u4,-u4) + O4(L,-u2-u3+u4,-u3+u4,u4)
        contrib = Wq*Wwin*Osum
        exact = Nl==Nr
        pair = exact & (((u1==u2)&(u3==u4))|((u1==u4)&(u3==u2)))
        res['exact_pair'] += contrib[pair].sum()
        res['exact_nonpair'] += contrib[exact&~pair].sum()
        res['hne0'] += contrib[~exact].sum()
    pref = (1.0/math.pi**4)*(1.0/16.0)*(2*math.pi)**3*2.0
    return L, gate, {k: pref*v/norm4 for k,v in res.items()}

for T in [float(sys.argv[1])]:
    t0=time.time(); L,g,o = measure(T)
    print(f"T={int(T)} l={L:.3f} k2gate={g:.4f} exact_pair={o['exact_pair']:+.5f} "
          f"exact_nonpair={o['exact_nonpair']:+.6f} hne0={o['hne0']:+.6f} ({time.time()-t0:.0f}s)", flush=True)
