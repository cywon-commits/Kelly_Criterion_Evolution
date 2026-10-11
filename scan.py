from kelly import run
import numpy as np
p=0.6
g=lambda f: p*np.log(1+f)+(1-p)*np.log(1-f)
fs=np.linspace(0,0.6,601); d=0.01
v=fs[g(fs)+np.log(1-d)>0]; print("Kelly 0.2, viable window (single lineage, delta=.01):",v.min(),v.max())
for xi in [1,4,16,64]:
  for fx in [0.1,0.2,0.3,0.4,0.5]:
    h=run(xi=xi,fixed=fx,T=2000)
    print("xi",xi,"f",fx,"final density",round(h[-1][1],3), "t",h[-1][0])
