import sys, numpy as np
from robust import R0
from scipy.optimize import minimize_scalar
q,v,mu,fsim=map(float,sys.argv[1:5])
g=lambda f: R0(f,(v,mu),'vm',M=500,K=120)
fs=np.linspace(0.05,0.8,16); G=[g(f) for f in fs]; i=int(np.argmax(G))
r=minimize_scalar(lambda f:-g(f),bounds=(fs[max(i-1,0)],fs[min(i+1,15)]),method='bounded',options={'xatol':2e-3})
print(f"q={q}: sim f={fsim:.3f} | chain best response to measured (v={v:.3g}, mu_d={mu:.3g}) = {r.x:.3f}; R0 at sim f = {g(fsim):.3f}",flush=True)
