import numpy as np
def Pmc(f,v,p=0.6,d=0.01,e=0.0,wmin=0.1,wcap=2.0,n=2_000_000,seed=0,Tmax=100000):
    rng=np.random.default_rng(seed)
    lw=np.zeros(n); alive=np.ones(n,bool); succ=np.zeros(n,bool)
    u,dn=np.log1p(f)+np.log(1-d),np.log1p(-f)+np.log(1-d); lo,hi=np.log(wmin),np.log(wcap)
    idx=np.arange(n)
    for t in range(Tmax):
        m=idx.size
        if m==0: break
        X=rng.random(m)<p
        lw_=np.minimum(lw[idx]+np.where(X,u,dn),hi)
        die=(lw_<lo)|(rng.random(m)<e)
        rep=(~die)&(lw_>=hi)&(rng.random(m)<v)
        succ[idx[rep]]=True
        keep=~die&~rep
        lw[idx]=lw_; idx=idx[keep]
    return succ.mean()
