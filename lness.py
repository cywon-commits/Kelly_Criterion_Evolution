import numpy as np, sys
from wp1s import P0
from scipy.optimize import brentq
p,d,e,lo,hi=0.6,0.01,float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])
kw=dict(p=p,d=d,e=e,M=2500); K=0.2
def vr(f):
    g=lambda lv: 2*P0(f,10**lv,**kw)-1
    a,b=g(-14),g(0)
    if b<0 or a>0: return None
    return 10**brentq(g,-14,0,xtol=1e-6)
out=[]
for fr in np.arange(lo,hi,0.0025):
    v=vr(fr)
    if v is None: continue
    s=[2*P0(fr+dd,v,**kw)-1 for dd in (-0.01,-0.0025,0.0025,0.01)]
    if max(s)<0: out.append(round(fr/K,3))
print(f"e={e}: local ESS (f/K) = {out}",flush=True)
