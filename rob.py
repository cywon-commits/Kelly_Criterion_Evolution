from kelly import run
import numpy as np, sys
p=float(sys.argv[1]); d=float(sys.argv[2]); xi=int(sys.argv[3])
g=lambda f: p*np.log(1+f)+(1-p)*np.log(1-f)+np.log(1-d)
fs=np.linspace(0,0.99,9901); v=fs[g(fs)>0]
fk=2*p-1
h=np.array(run(p=p,delta=d,xi=xi,T=60000,f0=fk,seed=3))
fe=np.nanmean(h[-200:,2]) if h[-1,1]>0 else np.nan
print(f"p={p} d={d} xi={xi} Kelly={fk:.2f} window=[{v.min():.3f},{v.max():.3f}] f_evo={fe:.3f} ratio={fe/fk:.2f} rho={h[-1,1]:.2f}",flush=True)
