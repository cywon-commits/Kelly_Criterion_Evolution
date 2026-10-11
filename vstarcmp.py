import numpy as np
from scipy.optimize import brentq
from robust import R0
from family import Fam
F=Fam(0.2,1.0,0.01,0.1)
def vex(f,mu):
    g=lambda lv: R0(f,(10**lv,mu),'vm',M=500,K=150)-1
    if g(0)<0: return np.inf
    return 10**brentq(g,-12,0,xtol=1e-6)
for mu in [0.0,0.003,0.01,0.03]:
    fs=np.linspace(0.06,0.7,17)
    ve=np.array([vex(f,mu) for f in fs]); vr=np.array([F.vstar(f,mu) for f in fs])
    ok=np.isfinite(ve)&np.isfinite(vr)
    i=np.argmin(np.where(ok,ve,np.inf)); j=np.argmin(np.where(ok,vr,np.inf))
    print('mu=%.3f argmin exact f=%.3f renewal f=%.3f | v*min exact %.3g renewal %.3g | max rel err v* %.2f'%(mu,fs[i],fs[j],ve[i],vr[j],np.max(np.abs(vr[ok]/ve[ok]-1))),flush=True)
