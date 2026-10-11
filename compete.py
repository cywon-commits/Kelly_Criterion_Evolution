from kelly import run
import numpy as np, sys
p,d,e,c1,c2,T=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),int(sys.argv[6])
fk=2*p-1; N=64*64; rng=np.random.default_rng(int(1e3*c1+7*c2))
res=[]
for rep in range(3):
    fi=np.where(rng.random(N)<0.5,c1*fk,c2*fk)
    h=np.array(run(p=p,delta=d,e=e,T=T,seed=rep+100,mf=True,sigma=0.0,f_init=fi,track=c1*fk))
    res.append(h[-1,2] if h[-1,1]>0 else np.nan)
print(f"p={p} d={d} e={e}: {c1}K vs {c2}K -> final share of {c1}K: {[round(r,2) for r in res]}",flush=True)
