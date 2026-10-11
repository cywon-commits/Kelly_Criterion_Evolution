import numpy as np, pickle
import xhybrid, xanal
xanal.CPI=0.74
mus=np.r_[0,np.logspace(-5,np.log10(0.0418),26)]
A=[];B=[]
for mu in mus:
    fa=xhybrid.fstar(mu,('pi','Tr','climb')); va=xhybrid.vstar(fa,mu,('pi','Tr','climb')) if np.isfinite(fa) else np.nan
    fb,vb=xanal.fstar(mu)
    A.append((mu,fa,va)); B.append((mu,fb,vb)); print(mu,round(fa,4),round(fb,4),flush=True)
# edge for theory A: smallest mu with min v*>=1
pickle.dump(dict(A=A,B=B),open('xcurves.pkl','wb'))
