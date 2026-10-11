import sys, numpy as np
from robust import R0
from scipy.optimize import brentq, minimize_scalar
w=float(sys.argv[1])
def mres(f):
    g=lambda lm: R0(f,10**lm,'lottery',wmin=w,M=500,K=100)-1
    if g(-8)<0: return -1
    return 10**brentq(g,-8,-1e-6,xtol=1e-6) if g(-1e-6)<0 else 1
fs=np.linspace(0.1,0.99,19); Z=[mres(f) for f in fs]; i=int(np.argmax(Z))
r=minimize_scalar(lambda f:-mres(f),bounds=(fs[max(i-1,0)],fs[min(i+1,18)]),method='bounded',options={'xatol':2e-3})
print(f"lottery wmin={w}: ESS f*={r.x:.3f} mu_r={mres(r.x):.3e}  | mu_r(f) grid:",' '.join(f"{a:.2f}:{b:.3f}" for a,b in zip(fs[::3],Z[::3])),flush=True)
