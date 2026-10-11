import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm
def lnparams(m, var=1.0):
    s2=np.log(1+var/(1+m)**2); return np.log(1+m)-s2/2, np.sqrt(s2)
def kelly(m,var=1.0,K=4000):
    mu,s=lnparams(m,var); R=np.exp(mu+s*norm.ppf((np.arange(K)+0.5)/K))-1
    return brentq(lambda f: np.mean(R/(1+f*R)),1e-6,0.999)
# ---------- simulation ----------
def run(L=64, m=0.2, var=1.0, delta=0.01, e=0.0, T=100000, sigma=0.01, wmin=0.1, wcap=2.0, f0=0.2, seed=0, mf=True, f_init=None, track=None, fixed=None, lottery=False, eramp=None):
    rng=np.random.default_rng(seed); N=L*L; mu,s=lnparams(m,var)
    occ=np.ones(N,bool); w=np.ones(N); f=np.full(N,f0 if fixed is None else fixed)
    if f_init is not None: f=np.asarray(f_init,float).copy()
    idx=np.arange(N); x=idx%L; y=idx//L
    nb=np.stack([((x+1)%L)+y*L,((x-1)%L)+y*L,x+((y+1)%L)*L,x+((y-1)%L)*L],1)
    hist=[]
    for t in range(T):
        if eramp is not None: e=eramp[0]+(eramp[1]-eramp[0])*t/T
        R=np.exp(mu+s*rng.standard_normal(N))-1
        w=np.where(occ,np.minimum(w*(1+f*R)*(1-delta),wcap),0)
        dead=occ&((w<wmin)|(rng.random(N)<e)); occ[dead]=False; w[dead]=0
        par=np.flatnonzero(occ&(w>=wcap))
        if par.size:
            par=rng.permutation(par)
            tgt=rng.integers(0,N,par.size) if mf else nb[par,rng.integers(0,4,par.size)]
            ok=(~occ[tgt]) if not lottery else (tgt!=par); par,tgt=par[ok],tgt[ok]
            _,fi=np.unique(tgt,return_index=True); par,tgt=par[fi],tgt[fi]
            w[par]/=2; w[tgt]=w[par]; occ[tgt]=True
            f[tgt]=f[par] if (fixed is not None or sigma==0) else np.clip(f[par]+sigma*rng.standard_normal(par.size),0,1)
        n=occ.sum()
        if n==0: hist.append((t,0,np.nan,np.nan,0)); break
        if t%50==0: hist.append((t,n/N,f[occ].mean() if track is None else np.mean(np.isclose(f[occ],track)),f[occ].var(),dead.sum()))
    return hist
# ---------- chain theory ----------
def P0(f,v,m=0.2,var=1.0,d=0.01,e=0.0,wmin=0.1,wcap=2.0,M=2000,K=300):
    mu,s=lnparams(m,var); lo,hi=np.log(wmin),np.log(wcap); x=np.linspace(lo,hi,M); h=x[1]-x[0]
    R=np.exp(mu+s*norm.ppf((np.arange(K)+0.5)/K))-1; steps=np.log1p(f*R)+np.log(1-d); c=(1-e)/K
    y=(x[:,None]+steps[None,:]).ravel(); I=np.repeat(np.arange(M),K)
    cap=y>=hi; ins=(y>=lo)&~cap
    b=np.bincount(I[cap],minlength=M)*c*v
    u=(y[ins]-lo)/h; k=np.minimum(u.astype(int),M-2); t=u-k
    rows=np.concatenate([I[ins],I[ins],I[cap]]); cols=np.concatenate([k,k+1,np.full(cap.sum(),M-1)])
    vals=np.concatenate([c*(1-t),c*t,np.full(cap.sum(),c*(1-v))])
    A=sp.csr_matrix((vals,(rows,cols)),shape=(M,M))
    P=spl.spsolve(sp.eye(M,format='csr')-A,b)
    return np.interp(np.log(wcap/2),x,P)
def vres(f,**kw):
    g=lambda lv: 2*P0(f,10**lv,**kw)-1
    a,b=g(-14),g(0)
    if b<0 or a>0: return None
    return 10**brentq(g,-14,0,xtol=1e-7)
def local_ess(frs,**kw):
    out=[]
    for fr in frs:
        v=vres(fr,**kw)
        if v is None: out.append((fr,None,None)); continue
        sL=2*P0(fr-0.004,v,**kw)-1; sR=2*P0(fr+0.004,v,**kw)-1
        out.append((fr,v,(sL,sR)))
    return out
