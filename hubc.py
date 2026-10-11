import numpy as np, sys
from cont import lnparams
def wheels(M=160,R=24):
    adj=[[] for _ in range(M*(R+1))]
    def add(a,b): adj[a].append(b); adj[b].append(a)
    for k in range(M):
        h=k*(R+1)
        for j in range(R):
            r=h+1+j; add(h,r); add(r,h+1+(j+1)%R)
        kk=((k+1)%M)*(R+1)
        for j in (0,R//2): add(h+1+j, kk+1+(j+R//2)%R)
    D=max(map(len,adj)); N=len(adj)
    nb=np.full((N,D),-1); deg=np.array([len(a) for a in adj])
    for i,a in enumerate(adj): nb[i,:len(a)]=a
    ishub=np.zeros(N,bool); ishub[::R+1]=True
    return nb,deg,ishub
def run(rule, e=0.0, p=0.6, m=0.2, var=1.0, fh0=0.3, fr0=0.3, delta=0.01, T=100000, sigma=0.01, wmin=0.1, wcap=2.0, seed=0):
    rng=np.random.default_rng(seed); nb,deg,ishub=wheels(); N=len(deg)
    occ=np.ones(N,bool); w=np.ones(N); G=np.zeros((N,2)); G[:,0]=fh0; G[:,1]=fr0; lm,ls=lnparams(m,var)  # G[:,0]=f_hub, G[:,1]=f_rim
    rec=[]
    for t in range(T):
        f=np.where(ishub,G[:,0],G[:,1])
        X=np.exp(lm+ls*rng.standard_normal(N))-1
        w=np.where(occ,np.minimum(w*(1+f*X)*(1-delta),wcap),0)
        dead=occ&((w<wmin)|(rng.random(N)<e)); occ[dead]=False; w[dead]=0
        par=np.flatnonzero(occ&(w>=2))
        if par.size:
            par=rng.permutation(par)
            if rule==1:
                tgt=nb[par,(rng.random(par.size)*deg[par]).astype(int)]
            else:
                cand=nb[par]; emp=(cand>=0)&~occ[np.where(cand>=0,cand,0)]
                sc=np.where(emp,rng.random(cand.shape),-1); k=sc.argmax(1)
                tgt=np.where(emp.any(1),cand[np.arange(par.size),k],par)
            ok=~occ[tgt]; par,tgt=par[ok],tgt[ok]
            _,fi=np.unique(tgt,return_index=True); par,tgt=par[fi],tgt[fi]
            w[par]/=2; w[tgt]=w[par]; occ[tgt]=True
            G[tgt]=np.clip(G[par]+sigma*rng.standard_normal((par.size,2)),0,1)
        if occ.sum()==0: return "EXTINCT",t
        if t>=0.8*T and t%100==0:
            rec.append((G[occ,0].mean(),G[occ,1].mean(),G[occ&ishub,0].mean(),occ[ishub].mean(),occ[~ishub].mean()))
    r=np.array(rec).mean(0)
    return dict(fhub=r[0],frim=r[1],fhub_expr=r[2],rho_hub=r[3],rho_rim=r[4])
rule=int(sys.argv[1]); e=float(sys.argv[2]); s=int(sys.argv[3]); fh0=float(sys.argv[4]); T=int(sys.argv[5])
out=run(rule,e,seed=s,fh0=fh0,fr0=0.3 if e>0 else 0.12,T=T)
print('fh0',fh0,end=' ')
print(f"rule{rule} e={e} seed{s}:", {k:round(v,2) for k,v in out.items()} if isinstance(out,dict) else out, flush=True)
