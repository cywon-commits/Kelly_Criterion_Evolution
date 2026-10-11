import numpy as np, sys
from cont import P0, vres, lnparams
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm
m,var,d=map(float,sys.argv[1:4]); kw=dict(m=m,var=var,d=d,e=0.0,M=1500,K=200)
mu,s=lnparams(m,var); R=np.exp(mu+s*norm.ppf((np.arange(4000)+0.5)/4000))-1
g=lambda f: np.mean(R/(1+f*R)); K=brentq(g,1e-6,0.999) if g(0.999)<0 else 1.0
res=[]
for v in (1e-3,1e-5):
    fs=np.linspace(0.005,min(0.95,1.5*K+0.1),30); P=[P0(f,v,**kw) for f in fs]; i=int(np.argmax(P))
    r=minimize_scalar(lambda f:-P0(f,v,**kw),bounds=(fs[max(i-1,0)],fs[min(i+1,29)]),method='bounded',options={'xatol':2e-4}); res.append(r.x)
D=np.log(20); a=np.log(2)
def F(f):
    L=np.log1p(f*R); mu_=L.mean()+np.log(1-d); s2=L.var(); lam=2*mu_/s2
    return (np.exp(lam*D)-np.exp(lam*a))/(np.sqrt(s2)*lam) if mu_>0 else -1
fs=np.linspace(0.002,0.95,3000); fd=fs[np.argmax([F(f) for f in fs])]
vr=vres(res[1],**kw)
print(f"m={m} var={var} d={d}: Kelly={K:.3f} | 2d/m={2*d/m:.3f} diffusion={fd:.3f} chain(v=1e-3,1e-5)={res[0]:.3f},{res[1]:.3f} c={res[1]/K:.3f} v_r={vr if vr is None else f'{vr:.1e}'}",flush=True)
