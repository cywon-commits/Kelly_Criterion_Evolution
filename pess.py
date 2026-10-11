import numpy as np, sys, pickle, time
from chainT import solve
from scipy.optimize import brentq
e=float(sys.argv[1]); fs=np.arange(float(sys.argv[2]),float(sys.argv[3]),0.005)
kw=dict(d=0.01,e=e,M=1200,K=150)
out=[]; t0=time.time()
for f in fs:
    g=lambda lv: 2*solve(f,10**lv,**kw)[0]-1
    try:
        if g(0)<0: continue
        out.append((f,brentq(g,-16,0,xtol=1e-6)))
    except Exception: pass
out=np.array(out); pickle.dump(out,open(f"out/pess_{e}.pkl","wb"))
i=np.argmin(out[:,1]); print(f"e={e}: argmin v_r at f={out[i,0]:.3f} (log10 v_r={out[i,1]:.3f}); {time.time()-t0:.0f}s",flush=True)
