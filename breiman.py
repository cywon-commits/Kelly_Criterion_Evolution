import numpy as np
from robust import kernel, interp_row
from scipy.optimize import minimize_scalar
def climb(f,m=0.2,var=1.0,d=0.01,wmin=0.1,wcap=2.0,start=1.0,M=600,K=150,mu=0.0):
    x,T=kernel(f,m,var,d,wmin,wcap,M,K); T=(1-mu)*T
    Q=T[:-1,:-1]; r=T[:-1,-1]; I=np.eye(M-1)
    s=interp_row(x,np.log(start))[:-1]
    h=np.linalg.solve(I-Q,r)          # prob reach cap
    th=np.linalg.solve(I-Q,h)         # E[tau; reach]
    Pi=s@h; tbar=(s@th)/Pi
    return Pi,tbar
if __name__=='__main__':
    for wmin in [1e-3,0.02,0.1]:
        r=minimize_scalar(lambda f: climb(f,wmin=wmin)[1],bounds=(0.1,1),method='bounded',options={'xatol':1e-3})
        g=minimize_scalar(lambda f: -(2*climb(f,wmin=wmin)[0]-1)/climb(f,wmin=wmin)[1],bounds=(0.1,1),method='bounded',options={'xatol':1e-3})
        print('wmin',wmin,'argmin tbar=%.3f'%r.x,'  argmax (2Pi-1)/tbar=%.3f'%g.x)
