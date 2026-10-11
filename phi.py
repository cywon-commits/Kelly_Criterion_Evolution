import numpy as np, re, pickle
from robust import kernel, interp_row, env_res
from rob2 import ess
def phi(f,v,mu,e,m=0.2,var=1.0,d=0.01,wmin=0.1,wcap=2.0,M=500,K=150):
    x,T=kernel(f,m,var,d,wmin,wcap,M,K); s=(1-e)*(1-mu)
    half=interp_row(x,x[-1]-np.log(2))
    A=T.copy(); Acap=A[:,-1].copy(); A[:,-1]=Acap*(1-v); A+=np.outer(Acap*v,half)
    A=s*A
    occ=np.linalg.solve((np.eye(M)-A).T, half)   # visits before each step
    ruin=occ@(s*(1-T.sum(1)))                     # survive killing, then fall below wmin
    kill=occ.sum()*(1-s)
    return kill/(kill+ruin), occ.sum()
rows=[]
for L in open('out/mixA.log').read().splitlines()+open('out/mixB.log').read().splitlines():
    a=float(re.search(r'alpha=([\de.-]+)',L).group(1)); f=float(re.search(r'f\*=([\d.]+)',L).group(1)); z=float(re.search(r'z_r=([\d.]+)',L).group(1))
    v=np.exp(-(1-a)*z); mu=1-np.exp(-a*z); p,Lf=phi(f,v,mu,0.0); rows.append(('alpha',a,f,p)); print('alpha',a,f,round(p,4),flush=True)
for e in [1e-4,3e-4,1e-3,3e-3,0.005,0.01,0.02,0.03]:
    kw=dict(m=0.2,var=1.0,d=0.01,wmin=0.1,wcap=2.0,e=e)
    f=ess('vacancy',kw,0.05,0.6); v=env_res(f,'vacancy',**kw,M=400,K=120)
    p,Lf=phi(f,v,0.0,e); rows.append(('e',e,f,p)); print('e',e,f,v,round(p,4),flush=True)
for e in [0.005,0.01,0.02]:
    kw=dict(m=0.2,var=1.0,d=0.01,wmin=0.1,wcap=2.0,e=e)
    f=ess('lottery',kw,0.2,1.0); mu=env_res(f,'lottery',**kw,M=400,K=120)
    p,Lf=phi(f,1.0,mu,e); rows.append(('lot',e,f,p)); print('lottery e',e,f,mu,round(p,4),flush=True)
pickle.dump(rows,open('phi.pkl','wb'))
