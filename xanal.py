import numpy as np
from functools import lru_cache
from scipy.optimize import brentq
from qsd import qsd
from xtheory import mom
a=np.log(2); b=np.log(1.0/0.1)   # start w=1: cap a above, ruin b below
CPI=0.72
def climbP(g,s2,h):
    D_=np.sqrt(g*g+2*h*s2); rp=(-g+D_)/s2; rm=(-g-D_)/s2
    M=np.array([[np.exp(rp*a),np.exp(rm*a)],[np.exp(-rp*b),np.exp(-rm*b)]]); return np.linalg.solve(M,[1,0]).sum()
@lru_cache(None)
def ing(f):
    q=qsd(f); return q['g'],q['s2'],CPI*q['pi'],q['Tr']
def vstar(f,mu):
    g,s2,pi,Tr=ing(round(f,5)); h=mu+1/Tr
    c=1/(2*climbP(g,s2,mu))      # climb: killed by extrinsic mortality; ruin handled by boundary b
    if c>=1: return np.inf
    return h*c/(h+pi*(1-c))
def fstar(mu,lo=0.04,hi=0.9):
    fs=np.linspace(lo,hi,88); vs=np.array([vstar(f,mu) for f in fs]); i=int(np.argmin(vs))
    if not np.isfinite(vs[i]): return np.nan, np.inf
    j=min(max(i,2),85); c=np.polyfit(fs[j-2:j+3],np.log(vs[j-2:j+3]),2); fx=-c[1]/(2*c[0]); return fx, vstar(fx,mu)
if __name__=='__main__':
    for mu in [0,1e-4,3e-4,1e-3,3e-3,0.005,0.01,0.02,0.03,0.035,0.04,0.045]:
        print(mu, [round(x,4) for x in fstar(mu)])
