import numpy as np, sys, pickle
from chainT import solve, vres
d=float(sys.argv[1]); B=pickle.load(open(f"out/bgrid_{d}.pkl","rb")); out=[]
for f,u,beta,v in B:
    h=4e-3; P0,T=solve(f,v,d=d); Pp=solve(f+h,v,d=d)[0]; Pm=solve(f-h,v,d=d)[0]
    dP=(Pp-Pm)/(2*h); d2P=(Pp-2*P0+Pm)/h**2
    out.append((f,u,beta,v,(2*d2P-4*dP**2)/T))
pickle.dump(np.array(out),open(f"out/cgrid_{d}.pkl","wb")); print(d,'done')
