import numpy as np, sys
from cont import P0, vres, kelly, lnparams
from scipy.optimize import minimize_scalar
from scipy.stats import norm
m,var,d,e=map(float,sys.argv[1:5])
kw=dict(m=m,var=var,d=d,e=e)
def best(v,lo=0.005,hi=0.95):
    fs=np.linspace(lo,hi,48); P=[P0(f,v,**kw) for f in fs]; i=int(np.argmax(P))
    r=minimize_scalar(lambda f:-P0(f,v,**kw),bounds=(fs[max(i-1,0)],fs[min(i+1,47)]),method='bounded',options={'xatol':1e-4}); return r.x
fr=2*d/m if e==0 else 0.3
for it in range(25):
    v=vres(fr,**kw)
    if v is None: fr*=1.2; continue
    b=best(v)
    if abs(b-fr)<3e-4: break
    fr=0.5*(fr+b)
# diffusion formula
mu,s=lnparams(m,var); R=np.exp(mu+s*norm.ppf((np.arange(4000)+0.5)/4000))-1
fs=np.linspace(0.002,0.95,4000); D=np.log(20); a=np.log(2)
def F(f):
    L=np.log1p(f*R); mu_=L.mean()+np.log(1-d); s2=L.var(); lam=2*mu_/s2
    return (np.exp(lam*D)-np.exp(lam*a))/(np.sqrt(s2)*lam) if mu_>0 else -1
fd=fs[np.argmax([F(f) for f in fs])]
K=kelly(m,var)
print(f"m={m} var={var} d={d} e={e}: Kelly={K:.3f} | 2d/m={2*d/m:.3f} diffusion={fd:.3f} chain ESS={b:.3f} (c={b/K:.3f}) v_r={v:.2e}",flush=True)
