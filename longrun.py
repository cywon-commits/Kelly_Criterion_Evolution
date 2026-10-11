from kelly import run
import numpy as np, sys
p,d,f0,T=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])
h=np.array(run(p=p,delta=d,f0=f0,T=T,seed=9,mf=True))
fk=2*p-1; k=len(h)//10
print(f"p={p} d={d} f0/K={f0/fk:.2f}: f/K traj", " ".join(f"{a:.2f}" for a in h[::k,2]/fk), f"| end {np.nanmean(h[-k:,2])/fk:.3f} rho {h[-1,1]:.3f}",flush=True)
