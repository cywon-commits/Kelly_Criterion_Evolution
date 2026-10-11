import numpy as np
from scipy.optimize import brentq, minimize_scalar
A=np.log(2)
def P(nu,mu,b,a=A):
    g=nu-nu**2/2; s2=nu**2
    D=np.sqrt(g*g+2*mu*s2); rp=(-g+D)/s2; rm=(-g-D)/s2   # e^{r x} solves s2/2 r^2 + g r - mu = 0
    # P(x=0) with P(a)=1, P(-b)=0: P = (e^{rp x}-c e^{rm x}) ...
    # solve u = C1 e^{rp x} + C2 e^{rm x}
    M=np.array([[np.exp(rp*a),np.exp(rm*a)],[np.exp(-rp*b),np.exp(-rm*b)]]); C=np.linalg.solve(M,[1,0]); return C.sum()
def mus(nu,b):
    f=lambda lm: P(nu,10**lm,b)-0.5
    if f(-8)<0: return 0
    return 10**brentq(f,-8,3)
def F(b):
    r=minimize_scalar(lambda n:-mus(n,b),bounds=(0.05,20),method='bounded',options={'xatol':1e-5}); return r.x, mus(r.x,b)
for b in [0.92,1.61,2.30,3.0,3.9,6,10,30]: print(b, F(b))
