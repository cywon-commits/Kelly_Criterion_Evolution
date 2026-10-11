import numpy as np, sys
def run(L=64, p=0.6, delta=0.01, xi=1, T=4000, sigma=0.01, wmin=0.1, wcap=2.0, f0=0.2, seed=0, fixed=None, e=0.0, thr=2.0, mf=False, ret_final=False, f_init=None, track=None):
    rng=np.random.default_rng(seed)
    N=L*L
    occ=np.ones(N,bool); w=np.ones(N); f=np.full(N,f0 if fixed is None else fixed)
    if f_init is not None: f=np.asarray(f_init,float).copy()
    idx=np.arange(N); x=idx%L; y=idx//L
    nb=np.stack([((x+1)%L)+y*L,((x-1)%L)+y*L,x+((y+1)%L)*L,x+((y-1)%L)*L],1)
    np_=L//xi if xi<=L else 1
    patch=(x//xi)%max(np_,1)+ (y//xi)%max(np_,1)*max(np_,1) if xi<L else np.zeros(N,int)
    hist=[]
    for t in range(T):
        X=np.where(rng.random(patch.max()+1)<p,1.0,-1.0)[patch]
        w=np.where(occ, w*(1+f*X)*(1-delta), 0)
        w=np.minimum(w,wcap)
        dead=occ&((w<wmin)|(rng.random(N)<e)); occ[dead]=False; w[dead]=0
        par=np.flatnonzero(occ&(w>=thr))
        if par.size:
            par=rng.permutation(par)
            tgt=rng.integers(0,N,par.size) if mf else nb[par,rng.integers(0,4,par.size)]
            ok=~occ[tgt]
            par,tgt=par[ok],tgt[ok]
            _,first=np.unique(tgt,return_index=True)
            par,tgt=par[first],tgt[first]
            w[par]/=2; w[tgt]=w[par]; occ[tgt]=True
            if fixed is None:
                f[tgt]=np.clip(f[par]+sigma*rng.standard_normal(par.size),0,1)
            else: f[tgt]=fixed
        n=occ.sum()
        if n==0: hist.append((t,0,np.nan)); break
        if t%50==0: hist.append((t,n/N,f[occ].mean()) if track is None else (t,n/N,np.mean(np.isclose(f[occ],track))))
    return (hist, f[occ].copy()) if ret_final else hist
