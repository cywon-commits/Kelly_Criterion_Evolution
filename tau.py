import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from cont import lnparams
def build(f,m=0.2,var=1.0,d=0.01,wmin=0.1,wcap=2.0,M=1500,K=200,ruin=True):
    mu,s=lnparams(m,var); lo,hi=np.log(wmin),np.log(wcap); x=np.linspace(lo,hi,M); h=x[1]-x[0]
    R=np.exp(mu+s*norm.ppf((np.arange(K)+0.5)/K))-1; steps=np.log1p(f*R)+np.log(1-d)
    y=(x[:,None]+steps[None,:]).ravel(); I=np.repeat(np.arange(M),K); c=1.0/K
    cap=y>=hi
    if not ruin: y=np.maximum(y,lo)
    ins=(y>=lo)&~cap
    u=(y[ins]-lo)/h; k=np.minimum(u.astype(int),M-2); t=u-k
    A=sp.csr_matrix((np.concatenate([c*(1-t),c*t,np.full(cap.sum(),c)]),(np.concatenate([I[ins],I[ins],I[cap]]),np.concatenate([k,k+1,np.full(cap.sum(),M-1)]))),shape=(M,M))
    hitcap=np.bincount(I[cap],minlength=M)*c   # prob next step lands at cap
    return x,A,hitcap,steps
def tau(f,e,**kw):
    x,A,hc,_=build(f,**kw); M=len(x)
    T=spl.spsolve((sp.eye(M,format='csr')-(1-e)*A).tocsc(),(1-e)*hc)   # expected discounted # cap-arrivals (= reproduction chances)
    return np.interp(np.log(2.0/2),x,T) if False else np.interp(np.log(kw.get('wcap',2.0)/2),x,T)
def fstar_tau(e,lo=0.02,hi=0.6,**kw):
    fs=np.linspace(lo,hi,30); T=[tau(f,e,**kw) for f in fs]; i=int(np.argmax(T))
    r=minimize_scalar(lambda f:-tau(f,e,**kw),bounds=(fs[max(i-1,0)],fs[min(i+1,29)]),method='bounded',options={'xatol':5e-4}); return r.x
