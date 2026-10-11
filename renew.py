import numpy as np, pickle
from breiman import climb
from scipy.optimize import minimize_scalar
res=pickle.load(open('lotscan2.pkl','rb')); out={}
for (m,V,w),r in res.items():
    if r[0]<0.11 and m<0.1: continue
    obj=lambda f: (lambda P,t: -(2*P-1)/(P*t))(*climb(f,m=m,var=V,wmin=w,M=500,K=150))
    lo=max(0.05,0.5*r[0]); hi=min(1.0,1.6*r[0])
    x=minimize_scalar(obj,bounds=(lo,hi),method='bounded',options={'xatol':2e-3}).x
    out[(m,V,w)]=(r[0],x); print((m,V,w),'chain %.3f renewal %.3f'%(r[0],x),flush=True)
pickle.dump(out,open('renew.pkl','wb'))
