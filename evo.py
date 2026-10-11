from kelly import run
import numpy as np, sys, pickle
xi=int(sys.argv[1]); T=int(sys.argv[2])
res={}
for s in range(2):
    h=run(xi=xi,T=T,sigma=0.01,seed=s)
    res[s]=h
    arr=np.array(h)
    print("xi",xi,"seed",s,"end t",int(arr[-1,0]), " f(t):", " ".join(f"{a[2]:.3f}" for a in arr[::len(arr)//10]), " rho(t):"," ".join(f"{a[1]:.2f}" for a in arr[::len(arr)//10]),flush=True)
pickle.dump(res,open(f"evo_xi{xi}.pkl","wb"))
