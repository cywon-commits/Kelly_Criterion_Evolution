import numpy as np, sys, time
from wp1s import P0, vres
from scipy.optimize import brentq, minimize_scalar
def best(v,kw,lo,hi):
    fs=np.linspace(lo,hi,41); Ps=np.array([P0(f,v,**kw) for f in fs]); i=Ps.argmax()
    a,b=fs[max(i-1,0)],fs[min(i+1,40)]
    r=minimize_scalar(lambda f:-P0(f,v,**kw),bounds=(a,b),method='bounded',options={'xatol':2e-5})
    return r.x
def ess(kw,lo=0.003,hi=0.9):
    def G(fr):
        v=vres(fr,**kw)
        if v is None: return np.nan
        return best(v,kw,lo,hi)-fr
    # bracket on grid of residents
    frs=np.linspace(lo+0.002,hi-0.05,24); gs=[G(f) for f in frs]
    out=[]
    for i in range(len(frs)-1):
        if np.isfinite(gs[i]) and np.isfinite(gs[i+1]) and gs[i]>0>gs[i+1]:
            out.append(brentq(G,frs[i],frs[i+1],xtol=2e-5))
    return out
if __name__=="__main__":
    p,d,e,M=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])
    t=time.time(); r=ess(dict(p=p,d=d,e=e,M=M))
    print(f"p={p} d={d} e={e}: ESS f/K = {[round(x/(2*p-1),3) for x in r]}  ({time.time()-t:.0f}s)",flush=True)
