from cont import run
import numpy as np, sys
m,var,d,e,f0,T,sig=map(float,sys.argv[1:8]); T=int(T)
h=np.array(run(m=m,var=var,delta=d,e=e,f0=f0,T=T,sigma=sig,seed=int(f0*1000)+int(e*1e4)))
k=max(len(h)//10,1)
print(f"m={m} var={var} d={d} e={e} f0={f0} sig={sig}: f traj "+" ".join(f"{a:.3f}" for a in h[::k,2])+f" | end {np.nanmean(h[-k:,2]):.4f} rho {np.mean(h[-k:,1]):.3f}",flush=True)
