import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from scipy.optimize import brentq, minimize_scalar
def P0(f,v,p=0.6,d=0.01,e=0.0,wmin=0.1,wcap=2.0,M=4000):
    lo,hi=np.log(wmin),np.log(wcap); x=np.linspace(lo,hi,M); h=x[1]-x[0]
    rows=[];cols=[];vals=[]; b=np.zeros(M); I=np.arange(M)
    for step,pr in ((np.log1p(f)+np.log(1-d),p),(np.log1p(-f)+np.log(1-d),1-p)):
        y=x+step; c=pr*(1-e)
        cap=y>=hi; ins=(y>=lo)&~cap
        b[cap]+=c*v; rows+=[I[cap]]; cols+=[np.full(cap.sum(),M-1)]; vals+=[np.full(cap.sum(),c*(1-v))]
        u=(y[ins]-lo)/h; k=np.minimum(u.astype(int),M-2); t=u-k
        rows+=[I[ins],I[ins]]; cols+=[k,k+1]; vals+=[c*(1-t),c*t]
    A=sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(M,M))
    P=spl.spsolve(sp.eye(M,format='csr')-A,b)
    return np.interp(0.0,x,P)
def vres(f,**kw):
    g=lambda v: 2*P0(f,v,**kw)-1
    if g(1.0)<0: return None
    return brentq(g,1e-10,1.0,xtol=1e-13)
def ess(f0,**kw):
    fr=f0
    for it in range(60):
        v=vres(fr,**kw)
        if v is None: return None,None
        r=minimize_scalar(lambda f:-P0(f,v,**kw),bounds=(0.002,0.98),method='bounded',options={'xatol':1e-5})
        if abs(r.x-fr)<1e-4: break
        fr=0.5*fr+0.5*r.x
    return r.x,v
