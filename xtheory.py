import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from cont import lnparams
m,V,d,wmin,wcap=0.2,1.0,0.01,0.1,2.0
D=np.log(wcap/wmin); a=np.log(2)
mu,s=lnparams(m,V); u=norm.ppf((np.arange(4000)+0.5)/4000); R=np.exp(mu+s*u)-1
def mom(f):
    x=np.log1p(f*R)+np.log(1-d); return x.mean(), x.var()
def spitzer(g,s2,N=4000):
    n=np.arange(1,N+1); return np.exp(-np.sum(norm.cdf(-np.sqrt(n)*g/np.sqrt(s2))/n))
def ingredients(f):
    g,s2=mom(f); lam=2*g/s2
    Tr=(np.exp(lam*D)-1-lam*D)/(g*lam)          # diffusion: reflecting cap, absorbing ruin, start at cap
    tc=a/g                                      # climb from cap/2
    pi=spitzer(g,s2)                            # fraction of steps ending at cap (Lindley/Spitzer)
    return g,s2,lam,Tr,tc,pi
def vstar(f,e):
    g,s2,lam,Tr,tc,pi=ingredients(f)
    if g<=0: return np.inf
    h=e+1/Tr; kap=(np.sqrt(g*g+2*h*s2)-g)/s2; Pc=np.exp(-a*kap)
    # need Pc*W(v)=1/2 with W=v+(1-v)*pi*v/(pi*v+h)
    W=lambda v: v+(1-v)*pi*v/(pi*v+h)
    if Pc*W(1.0)<0.5: return np.inf
    from scipy.optimize import brentq
    return brentq(lambda v: Pc*W(v)-0.5,1e-14,1.0)
def fstar(e):
    fs=np.linspace(0.03,0.9,160); vs=[vstar(f,e) for f in fs]; i=int(np.argmin(vs))
    r=minimize_scalar(lambda f: np.log(vstar(f,e)),bounds=(fs[max(i-1,0)],fs[min(i+1,159)]),method='bounded',options={'xatol':1e-5})
    return r.x, vstar(r.x,e)
if __name__=='__main__':
    print('f_I',fstar(0)[0], 'Tr(f_I)',ingredients(fstar(0)[0])[3])
    for e in [1e-4,3e-4,1e-3,3e-3,0.005,0.01,0.02,0.03,0.035,0.04,0.045,0.05]:
        f,v=fstar(e); print(e,round(f,3),'v*=%.3g'%v)
