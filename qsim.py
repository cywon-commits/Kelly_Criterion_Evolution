import numpy as np, sys
from cont import lnparams
def run(q, m=0.2, var=1.0, delta=0.01, T=60000, sigma=0.005, wmin=0.1, wcap=2.0, f0=0.2, seed=0, N=4096):
    rng=np.random.default_rng(seed); mu,s=lnparams(m,var)
    occ=np.ones(N,bool); w=np.ones(N); f=np.full(N,f0); rec=[]
    att=suc=disp=occsum=0
    for t in range(T):
        R=np.exp(mu+s*rng.standard_normal(N))-1
        w=np.where(occ,np.minimum(w*(1+f*R)*(1-delta),wcap),0)
        dead=occ&(w<wmin); occ[dead]=False; w[dead]=0
        par=np.flatnonzero(occ&(w>=wcap))
        if par.size:
            par=rng.permutation(par); tgt=rng.integers(0,N,par.size)
            _,fi=np.unique(tgt,return_index=True); keep=np.zeros(par.size,bool); keep[fi]=True
            ok=keep&(tgt!=par)&((~occ[tgt])|(rng.random(par.size)<q))
            nd=(ok&occ[tgt]).sum()
            if t>T*0.7: att+=par.size; suc+=ok.sum(); disp+=nd; occsum+=occ.sum()
            par,tgt=par[ok],tgt[ok]
            w[par]/=2; w[tgt]=w[par]; occ[tgt]=True
            f[tgt]=np.clip(f[par]+sigma*rng.standard_normal(par.size),0,1)
        if t%50==0: rec.append(f[occ].mean())
    return np.array(rec), suc/att, disp/occsum*1.0
if __name__=="__main__":
    q=float(sys.argv[1]); f0=float(sys.argv[2])
    rec,v,mud=run(q,f0=f0,seed=int(q*1000+f0*100))
    k=len(rec)//10
    print(f"q={q} f0={f0}: traj "+" ".join(f"{a:.3f}" for a in rec[::k])+f" | end {rec[-k:].mean():.3f}  v={v:.3e} mu_d={mud:.3e}",flush=True)
