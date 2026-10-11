from cont import run
import numpy as np, sys
mode=sys.argv[1]
if mode=='evo':
    for f0 in [0.3,0.42]:
        h=np.array(run(m=0.1,var=1.0,delta=0.005,e=0.01,f0=f0,T=60000,sigma=0.005,seed=int(f0*100)))
        k=len(h)//10; print(f"evo f0={f0}: end t={int(h[-1,0])} f={np.nanmean(h[-k:,2]):.3f} rho={np.mean(h[-k:,1]):.3f}",flush=True)
else:
    for fx in [0.135,0.19,0.25,0.3,0.35,0.4]:
        h=np.array(run(m=0.1,var=1.0,delta=0.005,e=0.01,fixed=fx,T=20000,seed=5))
        print(f"fixed f={fx}: {'EXTINCT at t=%d'%h[-1,0] if h[-1,1]==0 else 'survives rho=%.3f'%np.mean(h[-20:,1])}",flush=True)
