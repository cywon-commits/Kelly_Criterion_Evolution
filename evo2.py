from kelly import run
import numpy as np, sys, pickle
xi=int(sys.argv[1]); T=int(sys.argv[2]); f0=float(sys.argv[3]); d=float(sys.argv[4])
h=run(xi=xi,T=T,sigma=0.01,seed=7,f0=f0,delta=d)
arr=np.array(h); pickle.dump(arr,open(f"evo2_xi{xi}_f{f0}_d{d}.pkl","wb"))
k=max(1,len(arr)//12)
print(f"xi{xi} f0{f0} d{d} end{int(arr[-1,0])} f:", " ".join(f"{a[2]:.3f}" for a in arr[::k]), "| rho:"," ".join(f"{a[1]:.2f}" for a in arr[::k]),flush=True)
