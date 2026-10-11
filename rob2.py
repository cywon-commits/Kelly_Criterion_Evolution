import numpy as np, pickle, sys
from robust import R0, env_res
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from cont import lnparams
from chainT import solve
from multiprocessing import Pool
KW=dict(M=400,K=120)
VAR={'base':{}, 'split30':{'phi':0.3}, 'thr1.5':{'theta':1.5}}
def ess(variant,kw,lo,hi):
    sgn=1 if variant=='vacancy' else -1
    def obj(f):
        r=env_res(f,variant,**kw,**KW)
        return sgn*np.log10(r) if r else (10 if sgn>0 else 10)
    fs=np.linspace(lo,hi,15); vals=[obj(f) for f in fs]; i=int(np.argmin(vals))
    a,b=fs[max(i-1,0)],fs[min(i+1,14)]
    return minimize_scalar(obj,bounds=(a,b),method='bounded',options={'xatol':2e-4}).x
def job(a):
    name,variant,e=a; kw=dict(VAR[name]); kw['e']=e
    lo,hi=(0.03,0.4) if variant=='vacancy' else (0.2,1.0)
    return a, ess(variant,kw,lo,hi)
def g(f,m=0.2,var=1.0,d=0.01):
    mu,s=lnparams(m,var); z=mu+s*norm.ppf((np.arange(4000)+0.5)/4000); return np.mean(np.log1p(f*(np.exp(z)-1)))+np.log(1-d)
if __name__=='__main__':
    es=[0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3]
    jobs=[(n,'vacancy',e) for n in VAR for e in es]+[(n,'lottery',e) for n in VAR for e in [0,0.01]]
    with Pool(2) as p: res=dict(p.map(job,jobs))
    pickle.dump(res,open('rob2.pkl','wb'))
    fK=minimize_scalar(lambda f:-g(f),bounds=(0,1),method='bounded').x
    from scipy.optimize import brentq
    f0=brentq(g,fK,0.999)
    print('fK',fK,'g=0 at',f0)
    for n in VAR:
        f0v=res[(n,'vacancy',0)]; Tr=solve(f0v,0.0,0.2,1.0,0.01,0.0,0.1,2.0,1500,150)[1]
        print(n,'Tr',Tr,'lottery',res[(n,'lottery',0)],res[(n,'lottery',0.01)])
        for e in es: fv=res[(n,'vacancy',e)]; print('  e=%g X=%.3g f=%.4f ratio=%.3f'%(e,e*Tr,fv,fv/f0v))
