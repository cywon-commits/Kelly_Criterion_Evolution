import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from cont import lnparams
cases=[(0.2,1,0.005,0.055),(0.2,1,0.01,0.109),(0.2,1,0.02,0.202),(0.1,1,0.005,0.095),(0.3,1,0.01,0.078),(0.1,0.25,0.005,0.104),(0.1,0.25,0.002,0.042),(0.2,0.25,0.01,0.110)]
D=np.log(20); a=np.log(2)
for m,V,d,chain in cases:
    mu,s=lnparams(m,V); R=np.exp(mu+s*norm.ppf((np.arange(20000)+0.5)/20000))-1
    fs=np.linspace(0.005,0.5,2000)
    lam=[];gam=[]
    for f in fs:
        L=np.log1p(f*R)+np.log(1-d); mu_=L.mean(); s2=L.var()
        lam.append(2*mu_/s2 if mu_>0 else -1)
        if mu_>0:
            h=lambda g: np.mean(np.exp(-g*L))-1
            hi=2*mu_/s2*3+5
            try: gam.append(brentq(h,1e-6,hi))
            except: gam.append(np.nan)
        else: gam.append(-1)
    lam=np.array(lam); gam=np.array(gam)
    s_=np.array([np.std(np.log1p(f*R)) for f in fs])
    Fl=np.where(lam>0,(np.exp(lam*D)-np.exp(lam*a))/(s_*lam),-1)
    Fg=np.where(gam>0,(np.exp(gam*D)-np.exp(gam*a))/(s_*gam),-1)
    print(f"m={m} V={V} d={d}: 2d/m={2*d/m:.3f} argmax lambda={fs[np.nanargmax(lam)]:.3f} argmax gamma(Lundberg)={fs[np.nanargmax(gam)]:.3f} | F_lambda={fs[np.argmax(Fl)]:.3f} F_gamma={fs[np.nanargmax(Fg)]:.3f} | chain={chain}")
print("--- overshoot-shifted boundaries: D -> D + k*s, a -> a + k2*s")
for k in [0.583,1.166]:
  errs=[]
  for m,V,d,chain in cases:
    mu,s=lnparams(m,V); R=np.exp(mu+s*norm.ppf((np.arange(20000)+0.5)/20000))-1
    fs=np.linspace(0.005,0.5,1500); best=-1e9; fb=None
    for f in fs:
        L=np.log1p(f*R)+np.log(1-d); mu_=L.mean(); sd=L.std()
        if mu_<=0: continue
        g=brentq(lambda g: np.mean(np.exp(-g*L))-1,1e-6,2*mu_/sd**2*3+5)
        Fv=(np.exp(g*(D+k*sd))-np.exp(g*(a+k*sd/2)))/(sd*g)
        if Fv>best: best,fb=Fv,f
    errs.append(f"{fb/chain:.3f}")
  print("k=",k," F_gamma(shifted)/chain:",errs)
