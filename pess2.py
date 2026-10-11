import numpy as np, sys
from chainT import solve
from scipy.optimize import brentq
m,d,e,lo,hi=map(float,sys.argv[1:6])
kw=dict(m=m,d=d,e=e,M=1000,K=120); out=[]
for f in np.arange(lo,hi,0.01):
    g=lambda lv: 2*solve(f,10**lv,**kw)[0]-1
    try:
        if g(0)<0: continue
        out.append((f,brentq(g,-16,0,xtol=1e-5)))
    except Exception: pass
out=np.array(out); i=np.argmin(out[:,1])
print(f"m={m} d={d} e={e}: ESS(argmin v_r) f*={out[i,0]:.3f} log v_r={out[i,1]:.2f}; viable f range {out[0,0]:.2f}-{out[-1,0]:.2f}",flush=True)
