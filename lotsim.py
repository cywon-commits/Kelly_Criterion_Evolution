from cont import run
import numpy as np, sys
f0=float(sys.argv[1]); e=float(sys.argv[2])
h=np.array(run(f0=f0,e=e,T=60000,sigma=0.005,seed=int(f0*100),lottery=True))
k=len(h)//10
print(f"lottery sim f0={f0} e={e}: traj "+" ".join(f"{a:.3f}" for a in h[::k,2])+f" | end {np.nanmean(h[-k:,2]):.3f} rho {h[-1,1]:.2f}",flush=True)
