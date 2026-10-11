import numpy as np, pickle, sys
from functools import lru_cache
from scipy.optimize import brentq, minimize_scalar
from robust import kernel, interp_row
from rob2 import g as gfun
class Fam:
    def __init__(s,m,V,d,wmin,wcap=2.0,M=500,K=150):
        s.m,s.V,s.d,s.wmin,s.wcap,s.M,s.K=m,V,d,wmin,wcap,M,K
        s.ing=lru_cache(None)(s._ing)
    def _ing(s,f):
        x,T=kernel(f,s.m,s.V,s.d,s.wmin,s.wcap,s.M,s.K)
        w,vl=np.linalg.eig(T.T); i=np.argmax(w.real); q=np.abs(vl[:,i].real); q/=q.sum()
        st=interp_row(x,np.log(s.wcap/2))[:-1]
        return q[-1], 1/(1-w.real[i]), x, T, st
    def Pi(s,f,mu):
        pi,Tr,x,T,st=s.ing(f); A=(1-mu)*T; Q=A[:-1,:-1]; r=A[:-1,-1]
        return st@np.linalg.solve(np.eye(s.M-1)-Q,r)
    def vstar(s,f,mu):
        pi,Tr,*_=s.ing(round(float(f),4)); h=mu+1/Tr; P=s.Pi(round(float(f),4),mu); c=1/(2*P)
        if c>=1: return np.inf
        return h*c/(h+pi*(1-c))
    def fstar(s,mu,lo=0.02,hi=1.0,n=33):
        fs=np.linspace(lo,hi,n); vs=np.array([s.vstar(f,mu) for f in fs]); i=int(np.argmin(vs))
        if not np.isfinite(vs[i]): return np.nan,np.inf
        r=minimize_scalar(lambda f: np.log(s.vstar(f,mu)),bounds=(fs[max(i-1,0)],fs[min(i+1,n-1)]),method='bounded',options={'xatol':1e-3})
        return r.x, s.vstar(r.x,mu)
    def kelly(s):
        return minimize_scalar(lambda f:-gfun(f,s.m,s.V,s.d),bounds=(0,1),method='bounded').x
def run(args):
    name,m,V,d,wmin=args; F=Fam(m,V,d,wmin)
    f0,_=F.fstar(0.0); Tr0=F.ing(round(f0,4))[1]
    # edge: bisection on log mu
    sc=m*m/V; lo,hi=1e-4*sc,5*sc
    for _ in range(18):
        mid=np.sqrt(lo*hi); f,v=F.fstar(mid,lo=0.1)
        if np.isfinite(v) and v<1: lo=mid
        else: hi=mid
    muc=lo; fmax,_=F.fstar(muc,lo=0.1)
    mus=np.r_[0,muc*np.logspace(-6,0,25)[:-1]*(1-1e-3),muc]
    rows=[(mu,)+tuple(F.fstar(mu,lo=0.02 if mu<0.3*muc else 0.1)) for mu in mus]
    out=dict(name=name,m=m,V=V,d=d,wmin=wmin,Q=m*m/(2*d*V),b=np.log(2/(2*wmin)),D=np.log(2/wmin),fK=F.kelly(),f0=f0,Tr0=Tr0,muc=muc,fmax=fmax,rows=rows)
    print(name,'Q=%.2f b=%.2f fmin=%.4f Tr0=%.3g muc=%.4f fmax=%.3f fK=%.3f'%(out['Q'],out['b'],f0,Tr0,muc,fmax,out['fK']),flush=True)
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    fams=[('base',0.2,1.0,0.01,0.1),('Q4',0.2,1.0,0.005,0.1),('Q1',0.2,1.0,0.02,0.1),('b3.0',0.2,1.0,0.01,0.05),('b1.6',0.2,1.0,0.01,0.2),('mV',0.1,0.25,0.01,0.1)]
    with Pool(2) as p: res=p.map(run,fams)
    pickle.dump(res,open('family.pkl','wb'))
