import numpy as np, sys, pickle, scipy.sparse as sp, scipy.sparse.linalg as spl
from tau import build, fstar_tau
from cont import kelly
m,V,d=map(float,sys.argv[1:4]); wmin=float(sys.argv[4]) if len(sys.argv)>4 else 0.1; kw=dict(m=m,var=V,d=d,wmin=wmin,M=1000,K=120)
f0=fstar_tau(0.0,lo=0.01,hi=0.8,**kw)
x,A,hc,_=build(f0,**kw); Tr=spl.spsolve((sp.eye(len(x),format='csr')-A).tocsc(),np.ones(len(x))); Tr0=np.interp(0.0,x,Tr)
K=1.0
out=[]
for X in [0.03,0.1,0.3,1,3,10,30,100,300]:
    e=X/Tr0
    if e>0.05: break
    out.append((X,e,fstar_tau(e,lo=0.01,hi=0.9,**kw)))
pickle.dump(dict(m=m,V=V,d=d,wmin=wmin,f0=f0,Tr0=Tr0,K=K,out=out),open(f"out/univ_{m}_{V}_{d}"+("" if wmin==0.1 else f"_w{wmin}")+".pkl","wb"))
print(f"m={m} V={V} d={d}: f0={f0:.3f} (2d/m={2*d/m:.3f}) Kelly={K:.3f} Tr0={Tr0:.3g} | "+" ".join(f"X={a:g}:{c:.3f}" for a,b,c in out),flush=True)
