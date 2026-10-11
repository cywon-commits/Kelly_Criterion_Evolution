import numpy as np
from stories import path
for s in range(60):
    rng=np.random.default_rng(s); Z=rng.standard_normal(800); vac=rng.random(800)<0.02
    a=path(0.11,Z,vac); b=path(0.6,Z,vac)
    if len(a[2])==2 and len(b[2])==0 and b[3] and b[3]>150 and b[3]<500: print('wait seed',s,a[2],b[3])
for s in range(60):
    rng=np.random.default_rng(1000+s); Z=rng.standard_normal(60); vac=np.ones(60,bool)
    a=path(0.11,Z,vac,60); b=path(0.6,Z,vac,60)
    if len(a[2])==1 and len(b[2])==2 and b[3] is None: print('disp seed',1000+s,a[2],b[2])
