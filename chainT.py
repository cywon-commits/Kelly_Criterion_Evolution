import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from scipy.stats import norm
from scipy.optimize import brentq
from cont import lnparams
def solve(f,v,m=0.2,var=1.0,d=0.01,e=0.0,wmin=0.1,wcap=2.0,M=1500,K=200):
    mu,s=lnparams(m,var); lo,hi=np.log(wmin),np.log(wcap); x=np.linspace(lo,hi,M); h=x[1]-x[0]
    R=np.exp(mu+s*norm.ppf((np.arange(K)+0.5)/K))-1; steps=np.log1p(f*R)+np.log(1-d); c=(1-e)/K
    y=(x[:,None]+steps[None,:]).ravel(); I=np.repeat(np.arange(M),K)
    cap=y>=hi; ins=(y>=lo)&~cap
    b=np.bincount(I[cap],minlength=M)*c*v
    u=(y[ins]-lo)/h; k=np.minimum(u.astype(int),M-2); t=u-k
    A=sp.csr_matrix((np.concatenate([c*(1-t),c*t,np.full(cap.sum(),c*(1-v))]),(np.concatenate([I[ins],I[ins],I[cap]]),np.concatenate([k,k+1,np.full(cap.sum(),M-1)]))),shape=(M,M))
    B=(sp.eye(M,format='csr')-A).tocsc(); lu=spl.splu(B)
    P=lu.solve(b); T=lu.solve(np.ones(M))
    x0=np.log(wcap/2); return np.interp(x0,x,P), np.interp(x0,x,T)
def vres(f,**kw):
    g=lambda lv: 2*solve(f,10**lv,**kw)[0]-1
    return 10**brentq(g,-16,0,xtol=1e-8)
def rates(f,**kw):
    v=vres(f,**kw); P,T=solve(f,v,**kw); h=2e-4
    dP=(solve(f+h,v,**kw)[0]-solve(f-h,v,**kw)[0])/(2*h)
    return dict(v=v,T=T,u=1/(2*T),beta=2*dP/T)
