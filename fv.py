import numpy as np, sys, pickle
from chainT import solve
from scipy.optimize import minimize_scalar
e=float(sys.argv[1]); kw=dict(d=0.01,e=e,M=1200,K=150); res=[]
for lv in np.linspace(-6,0,19):
    v=10**lv; fs=np.linspace(0.02,0.7,35); P=[solve(f,v,**kw)[0] for f in fs]; i=int(np.argmax(P))
    r=minimize_scalar(lambda f:-solve(f,v,**kw)[0],bounds=(fs[max(i-1,0)],fs[min(i+1,34)]),method='bounded',options={'xatol':1e-3})
    res.append((lv,r.x,-r.fun))
pickle.dump(np.array(res),open(f"out/fv_{e}.pkl","wb")); print("e",e," ".join(f"{a:.1f}:{b:.3f}" for a,b,c in res),flush=True)
