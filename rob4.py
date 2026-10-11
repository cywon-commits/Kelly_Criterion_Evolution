from kelly import run
import numpy as np, sys
e=float(sys.argv[1]); mode=sys.argv[2]
res=[]
for s in [21,22]:
    if mode=="evo": h=np.array(run(T=60000,f0=0.2,seed=s,e=e))
    else: h=np.array(run(T=20000,fixed=float(mode),seed=s,e=e))
    if h[-1,1]==0: res.append(f"X@{int(h[-1,0])}")
    else:
        q=h[-len(h)//5:]; res.append(f"rho{np.mean(q[:,1]):.2f},f{np.nanmean(q[:,2]):.3f}")
print(f"e={e} {mode}: {res}",flush=True)
