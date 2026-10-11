import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from xtheory import mom, D
def qsd(f,Deff=None):
    g,s2=mom(f); lam=2*g/s2; L=D if Deff is None else Deff
    if lam*L<2:
        h=lambda k: np.cos(k*L)*2*k - lam*np.sin(k*L)
        k=brentq(h,1e-9,np.pi/L-1e-12)
        rho=lambda y: np.exp(-lam*y/2)*(np.cos(k*y)-lam/(2*k)*np.sin(k*y)); k2=k*k
    else:
        h=lambda q: np.tanh(q*L)*lam-2*q
        k=brentq(h,1e-9,lam/2*(1-1e-12))
        rho=lambda y: np.exp(-lam*y/2)*(np.cosh(k*y)-lam/(2*k)*np.sinh(k*y)); k2=-k*k
    Z=quad(rho,0,L)[0]; theta=s2/2*(k2+lam*lam/4)
    return dict(rho0=1/Z, Tr=1/theta, pi=np.sqrt(s2/2)/Z, lam=lam,g=g,s2=s2)
if __name__=='__main__':
    from xhybrid import chain_ing
    for f in [0.1,0.2,0.3,0.4,0.5,0.6]:
        q=qsd(f); pc,Tc=chain_ing(f); print(f,'pi qsd %.3f chain %.3f | Tr qsd %.3g chain %.3g'%(q['pi'],pc,q['Tr'],Tc))
