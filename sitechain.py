import numpy as np
from robust import kernel, interp_row
def site(f,v,e,m=0.2,var=1.0,d=0.01,wmin=0.1,wcap=2.0,M=400,K=120):
    x,T=kernel(f,m,var,d,wmin,wcap,M,K); cap=np.zeros(M); cap[-1]=1
    # step: bet (T) with background survival (1-e); at cap reproduce w.p. v -> jump to cap/2
    half=interp_row(x,x[-1]-np.log(2))
    A=(1-e)*T.copy()
    # mass arriving at cap: fraction v moves to half
    Acap=A[:,-1].copy(); A[:,-1]=Acap*(1-v); A+=np.outer(Acap*v,half)
    s0=half
    occ=np.linalg.solve((np.eye(M)-A).T, s0)   # expected visits to each state (pre-step state)
    L=occ.sum()
    # fraction of steps that END at cap (ready at reproduction check)
    arrive=occ@((1-e)*T)[:,-1]
    return dict(pi=arrive/L, d=1/L, L=L)
if __name__=='__main__':
    print(site(0.354,0.94,0.01), site(0.288,0.157,0.01))
    print(site(0.297,0.0095,0.0), site(0.1185,0.0027,0.0))
