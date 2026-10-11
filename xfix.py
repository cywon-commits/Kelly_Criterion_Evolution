import numpy as np, pickle, xhybrid
from scipy.optimize import minimize_scalar
def fstar(mu,use=('pi','Tr','climb')):
    fs=np.linspace(0.05,0.8,31); vs=np.array([xhybrid.vstar(f,mu,use) for f in fs]); i=int(np.argmin(vs))
    if not np.isfinite(vs[i]): return np.nan
    r=minimize_scalar(lambda f: np.log(xhybrid.vstar(f,mu,use)),bounds=(fs[max(i-1,0)],fs[min(i+1,30)]),method='bounded',options={'xatol':5e-4})
    return r.x
d=pickle.load(open('xcurves.pkl','rb'))
A=[]
for mu,fa,va in d['A']:
    f=fstar(mu); A.append((mu,f)); print(mu,round(f,4),flush=True)
d['A']=A; pickle.dump(d,open('xcurves.pkl','wb'))
