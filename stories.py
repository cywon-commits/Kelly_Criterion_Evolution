import numpy as np
from cont import lnparams
mu,sd=lnparams(0.2,1.0); d=0.01; cap=np.log(2); lo=np.log(0.1)
def path(f,Z,vac,Tdeath=None):
    lw=0.0; xs=[0]; ys=[lw]; reps=[]; ruin=None
    for t,z in enumerate(Z,1):
        if Tdeath is not None and t>Tdeath: break
        lw=min(lw+np.log1p(f*(np.exp(mu+sd*z)-1))+np.log(1-d),cap)
        if lw>=cap and vac[t-1]: reps.append(t); xs.append(t); ys.append(cap); lw=cap-np.log(2)
        if lw<lo: ruin=t; xs.append(t); ys.append(lo); break
        xs.append(t); ys.append(lw)
    return np.array(xs),np.array(ys),reps,ruin
def mc(f,v,T,Tdeath=None,n=4000,seed=0):
    rng=np.random.default_rng(seed); cnt=[]
    for i in range(n):
        Z=rng.standard_normal(T); vac=rng.random(T)<v; _,_,r,_=path(f,Z,vac,Tdeath); cnt.append(len(r))
    return np.mean(cnt)
if __name__=="__main__":
    print('waiting v=0.02 T=1500: cautious',mc(0.11,0.02,1500,n=1500),' bold',mc(0.6,0.02,1500,n=1500))
    print('displace v=1, death at 60: cautious',mc(0.11,1.0,60,60,n=3000),' bold',mc(0.6,1.0,60,60,n=3000))
