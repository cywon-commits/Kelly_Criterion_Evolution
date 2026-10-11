from cont import run
import numpy as np, sys, pickle
L,T,seed=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
h=np.array(run(L=L,m=0.2,var=1.0,delta=0.005,f0=0.2,T=T,sigma=0.005,seed=seed))
pickle.dump(h,open(f"out/nsim_L{L}_s{seed}.pkl","wb")); print("done",L,seed,flush=True)
