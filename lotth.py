import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
from cont import lnparams
def xi(f,m,var,d,K=4000):
    mu,s=lnparams(m,var); R=np.exp(mu+s*norm.ppf((np.arange(K)+0.5)/K))-1
    return np.log1p(f*R)+np.log(1-d)
def roots(x,q):
    phi=lambda t: np.log(q)+np.log(np.mean(np.exp(t*x)))
    t0=minimize_scalar(phi,bounds=(-50,50),method='bounded').x
    hi=t0+0.5
    while phi(hi)<0: hi=t0+2*(hi-t0)
    lo=t0-0.5
    while phi(lo)<0: lo=t0-2*(t0-lo)
    return brentq(phi,t0,hi), brentq(phi,lo,t0)
def Pup(f,mu,m=0.2,var=1.0,d=0.01,wmin=0.1,wcap=2.0):
    x=xi(f,m,var,d); a=np.log(2); b=np.log(wcap/2/wmin); q=1-mu
    k1,k2=roots(x,q); A1,A2=np.exp(k1*a),np.exp(k2*a); B1,B2=np.exp(-k1*b),np.exp(-k2*b)
    return (B2-B1)/(A1*B2-A2*B1)
def mustar(f,**kw):
    g=lambda lm: Pup(f,10**lm,**kw)-0.5
    if g(-12)<0: return 0.0
    return 10**brentq(g,-12,-1e-6)
def ess(**kw):
    r=minimize_scalar(lambda f:-np.log(mustar(f,**kw)+1e-300),bounds=(0.05,0.99),method='bounded',options={'xatol':1e-4})
    return r.x, mustar(r.x,**kw)
if __name__=='__main__':
    print(ess())
    for w in [0.02,0.05,0.2,0.4]: print('wmin',w,ess(wmin=w))
    for d in [0.0,0.02]: print('d',d,ess(d=d))
