import numpy as np, pickle
from robust import R0
from family import Fam
F=Fam(0.2,1.0,0.01,0.1)
rows=[]
for f in [0.1,0.2,0.3,0.45,0.6]:
    pi,Tr,*_=F.ing(f)
    for mu in [0.0,1e-3,1e-2,3e-2]:
        Pi=F.Pi(f,mu)
        for v in [1e-3,1e-2,0.1,0.5,1.0]:
            Pex=R0(f,(v,mu),'vm',M=500,K=150)/2
            h=mu+1/Tr
            Pc=Pi*(v+(1-v)*pi*v/(pi*v+h))
            hd=1-(1-mu)*(1-1/Tr)
            Pd=Pi*(v+(1-v)*pi*v/(pi*v+hd*(1-pi*v)))
            rows.append((f,mu,v,Pex,Pc,Pd)); 
    print(f,flush=True)
pickle.dump(rows,open('p0grid.pkl','wb'))
a=np.array(rows); rc=np.abs(a[:,4]/a[:,3]-1); rd=np.abs(a[:,5]/a[:,3]-1)
print('cont: median %.3f max %.3f | disc: median %.3f max %.3f'%(np.median(rc),rc.max(),np.median(rd),rd.max()))
for r in a[np.argsort(-rc)][:8]: print(np.round(r,4))
