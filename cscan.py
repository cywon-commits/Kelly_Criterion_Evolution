import numpy as np, sys
from cont import P0, vres
e=float(sys.argv[1]); lo,hi=float(sys.argv[2]),float(sys.argv[3])
kw=dict(m=0.2,var=1.0,d=0.01,e=e,M=1200,K=150)
out=[]; G=[]
for fr in np.arange(lo,hi,0.005):
    v=vres(fr,**kw)
    if v is None: continue
    sR=2*P0(fr+0.005,v,**kw)-1; sL=2*P0(fr-0.005,v,**kw)-1
    G.append((round(fr,3),round(sL,4),round(sR,4)))
    if sL<0 and sR<0: out.append(round(fr,3))
print(f"e={e}: local ESS f = {out}; gradient signs:", [(g[0], '+' if g[2]>0 else '-') for g in G],flush=True)
