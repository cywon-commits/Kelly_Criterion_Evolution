import numpy as np
from functools import lru_cache
from scipy.optimize import brentq, minimize_scalar
from robust import kernel
from xtheory import ingredients
from breiman import climb
from chainT import solve
@lru_cache(None)
def chain_ing(f):
    x,T=kernel(f,0.2,1.0,0.01,0.1,2.0,500,150)
    w,vl=np.linalg.eig(T.T); i=np.argmax(w.real); q=np.abs(vl[:,i].real); q/=q.sum()
    pi=q[-1]; Tr=1/(1-w.real[i])   # quasi-stationary ruin rate
    return pi,Tr
def vstar(f,e,use):
    g,s2,lam,Tr,tc,pi=ingredients(f)
    if 'pi' in use or 'Tr' in use:
        pic,Trc=chain_ing(round(f,4))
        if 'pi' in use: pi=pic
        if 'Tr' in use: Tr=Trc
    h=e+1/Tr
    if 'climb' in use: Pc=climb(round(f,4),mu=e,M=500,K=150)[0]
    else:
        kap=(np.sqrt(g*g+2*h*s2)-g)/s2; Pc=np.exp(-np.log(2)*kap)
    W=lambda v: v+(1-v)*pi*v/(pi*v+h)
    if Pc*W(1.0)<0.5: return np.inf
    return brentq(lambda v: Pc*W(v)-0.5,1e-14,1.0)
def fstar(e,use):
    fs=np.linspace(0.05,0.8,31); vs=np.array([vstar(f,e,use) for f in fs]); i=int(np.argmin(vs))
    if not np.isfinite(vs[i]): return np.nan
    j=min(max(i,2),28); c=np.polyfit(fs[j-2:j+3],np.log(vs[j-2:j+3]),2); return -c[1]/(2*c[0])
if __name__=='__main__':
    import sys
    for use in [(),('pi',),('Tr',),('climb',),('pi','Tr','climb')]:
        print(use,[round(fstar(e,use),3) for e in [1e-3,0.005,0.01,0.02,0.03]],flush=True)
