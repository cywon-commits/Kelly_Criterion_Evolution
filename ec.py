import numpy as np, sys
from chainT import solve
from scipy.optimize import brentq
m,d=float(sys.argv[1]),float(sys.argv[2]); kw=dict(m=m,d=d,M=1000,K=120)
fs=np.arange(0.02,0.9,0.02); out=[]
for f in fs:
    g=lambda e: 2*solve(f,1.0,e=e,**kw)[0]-1
    if g(0)<0: out.append((f,0)); continue
    out.append((f,brentq(g,0,0.5,xtol=1e-6)))
out=np.array(out); i=np.argmax(out[:,1])
print(f"m={m} d={d}: e_c(f): "+" ".join(f"{a:.2f}:{b:.4f}" for a,b in out[::3])+f" | max e_c={out[i,1]:.4f} at f={out[i,0]:.2f}",flush=True)
np.save(f"out/ec_{m}_{d}.npy",out)
