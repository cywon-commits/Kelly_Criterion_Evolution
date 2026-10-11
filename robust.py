import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
from cont import lnparams
def kernel(f,m,var,d,wmin,wtop,M,K):
    mu,s=lnparams(m,var); lo,hi=np.log(wmin),np.log(wtop); x=np.linspace(lo,hi,M); h=x[1]-x[0]
    R=np.exp(mu+s*norm.ppf((np.arange(K)+0.5)/K))-1; st=np.log1p(f*R)+np.log(1-d)
    T=np.zeros((M,M))
    for j in range(K):
        y=np.minimum(x+st[j],hi); ok=y>=lo; u=(y[ok]-lo)/h; k=np.minimum(u.astype(int),M-2); t=u-k; I=np.nonzero(ok)[0]
        np.add.at(T,(I,k),(1-t)/K); np.add.at(T,(I,k+1),t/K)
    return x,T   # T: one betting step incl. cap clipping; rows lose mass = ruin
def interp_row(x,val):  # weights to place a point on grid
    M=len(x); h=x[1]-x[0]; u=(val-x[0])/h; k=int(min(max(u,0),M-1-1e-9)); t=u-k; w=np.zeros(M); w[k]+=1-t; w[k+1]+=t; return w
def R0(f,env,variant,m=0.2,var=1.0,d=0.01,e=0.0,wmin=0.1,wcap=2.0,phi=0.5,theta=None,M=500,K=120):
    """mean offspring (next-gen spectral radius). env = v (vacancy prob) or mu_d (displacement mortality)"""
    wtop=wcap; x,T=kernel(f,m,var,d,wmin,wtop,M,K); I=np.eye(M)
    thr=np.log(theta) if theta else x[-1]-1e-12
    canrep=(x>=thr-1e-12).astype(float)
    if variant=='lottery': v=1.0; surv=(1-e)*(1-env)
    elif variant=='vm':
        v,mu=env; surv=(1-e)*(1-mu)
    elif variant.startswith('mixed'):
        a=float(variant[5:]); v=np.exp(-(1-a)*env); surv=(1-e)*np.exp(-a*env)
    else: v=env; surv=(1-e)
    # step: bet (T) then with prob v reproduce if above threshold; staying = not reproducing
    A=surv*T*(1-v*canrep)[None,:]          # move to state j and not reproduce there
    Brep=surv*T*(v*canrep)[None,:]          # move to state j and reproduce there
    # reproduction at state j: parent -> x_j + ln(1-phi), child -> x_j + ln(phi)
    Pm=np.array([interp_row(x,xj+np.log(1-phi)) for xj in x]); Cm=np.array([interp_row(x,xj+np.log(phi)) for xj in x])
    N=np.linalg.solve(I-A,Brep)             # N[i,j]: expected # reproduction events at j starting from i (first one) -> actually prob of first rep at j
    G=N@(Pm+Cm)                             # next-gen: from start i, distribution of the two new starts
    return np.max(np.abs(np.linalg.eigvals(G)))
def env_res(f,variant,**kw):
    g=(lambda lv: R0(f,10**lv,variant,**kw)-1)
    if variant=='lottery':   # R decreasing in mu_d; find max tolerable mu_d
        if g(-12)<0: return None
        return 10**brentq(g,-12,-1e-9,xtol=1e-6) if g(-1e-9)<0 else 1.0
    else:
        if g(0)<0: return None
        return 10**brentq(g,-14,0,xtol=1e-6)
def ess(variant,lo,hi,n=21,**kw):
    fs=np.linspace(lo,hi,n); E=[]
    for f in fs:
        r=env_res(f,variant,**kw); E.append(np.nan if r is None else r)
    E=np.array(E)
    i=np.nanargmax(E) if variant=='lottery' else np.nanargmin(E)
    return fs,E,fs[i]
