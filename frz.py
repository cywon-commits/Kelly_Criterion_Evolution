from cont import run
import numpy as np, sys, pickle
m,V,d,f0,T,sig,seed=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),int(sys.argv[7])
h=np.array(run(m=m,var=V,delta=d,f0=f0,T=T,sigma=sig,seed=seed))
pickle.dump(h,open(f"out/frz_{m}_{d}_{f0}_{sig}_{seed}.pkl","wb")); print("done",flush=True)
