import numpy as np
from scipy.optimize import brentq, minimize_scalar
M=1500
def Pvec(f,v,p=0.6,d=0.01,e=0.0,wmin=0.1,wcap=2.0):
    lo,hi=np.log(wmin),np.log(wcap)
    x=np.linspace(lo,hi,M); h=x[1]-x[0]
    A=np.zeros((M,M)); b=np.zeros(M)
    for step,pr in ((np.log(1+f)+np.log(1-d),p),(np.log(1-f)+np.log(1-d) if f<1 else -np.inf,1-p)):
        y=x+step
        for i,yi in enumerate(y):
            c=pr*(1-e)
            if yi<lo: continue
            if yi>=hi:  # cap: reproduce w.p. v, else stay at cap
                b[i]+=c*v; A[i,M-1]+=c*(1-v); continue
            k=int((yi-lo)/h); t=(yi-lo)/h-k
            if k>=M-1: k,t=M-2,1.0
            A[i,k]+=c*(1-t); A[i,k+1]+=c*t
    P=np.linalg.solve(np.eye(M)-A,b)
    return x,P
def P0(f,v,**kw):
    x,P=Pvec(f,v,**kw); return np.interp(0.0,x,P)
def vres(f,**kw):
    g=lambda v: 2*P0(f,v,**kw)-1
    if g(1.0)<0: return None
    return brentq(g,1e-9,1.0,xtol=1e-12)
def ess(fk=0.2,**kw):
    # iterate: f_{n+1} = argmax_f P0(f, v(f_n))
    fr=fk
    for it in range(30):
        v=vres(fr,**kw)
        if v is None: return None,None
        r=minimize_scalar(lambda f:-P0(f,v,**kw),bounds=(0.005,0.95),method='bounded',options={'xatol':1e-4})
        if abs(r.x-fr)<2e-4: break
        fr=0.5*fr+0.5*r.x
    return r.x,v
if __name__=="__main__":
    for e in [0,0.002,0.005,0.01,0.015,0.02,0.03]:
        f,v=ess(e=e); print(f"e={e}: f_ESS/Kelly={f/0.2 if f else None:.3f}  vacancy v={v:.2e}",flush=True)
