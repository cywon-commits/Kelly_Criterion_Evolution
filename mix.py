import sys, numpy as np
from robust import R0
from scipy.optimize import brentq, minimize_scalar
a=float(sys.argv[1]); e=float(sys.argv[2]); var='mixed%g'%a
def zres(f):
    g=lambda z: R0(f,z,var,e=e,M=400,K=100)-1
    if g(0)<0: return -1
    hi=1.0
    while g(hi)>0 and hi<200: hi*=2
    return brentq(g,0,hi,xtol=1e-7)
fs=np.linspace(0.03,0.98,20); Z=[zres(f) for f in fs]; i=int(np.argmax(Z))
r=minimize_scalar(lambda f:-zres(f),bounds=(fs[max(i-1,0)],fs[min(i+1,19)]),method='bounded',options={'xatol':2e-3})
z=zres(r.x); print(f"alpha={a:g} e={e}: ESS f*={r.x:.3f}  z_r={z:.3f}  v={np.exp(-(1-a)*z):.2e} mu_density={1-np.exp(-a*z):.2e}",flush=True)
