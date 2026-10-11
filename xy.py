import numpy as np, pickle, glob, scipy.sparse as sp, scipy.sparse.linalg as spl
from tau import build
pts=[]
for fn in sorted(glob.glob('out/univ_*.pkl')):
    D=pickle.load(open(fn,'rb')); w=D.get('wmin',0.1); kw=dict(m=D['m'],var=D['V'],d=D['d'],wmin=w,M=1000,K=120)
    x,A,hc,_=build(D['f0'],ruin=False,**kw); M=len(x); A2=A.tolil(); A2[:,M-1]=0
    Tc=spl.spsolve((sp.eye(M,format='csr')-A2.tocsr()).tocsc(),np.ones(M)); tc=np.interp(np.log(1.0),x,Tc)
    lab=f"m={D['m']},V={D['V']},d={D['d']},w={w}"
    for X,e,f in D['out']: pts.append((lab,X,e*tc,f/D['f0'],D['Tr0'],tc))
pickle.dump(pts,open('out/xy.pkl','wb'))
for p in pts:
    if p[1]>=1: print(f"{p[0]:32s} X={p[1]:6g} Y={p[2]:.2e} ratio={p[3]:.3f} Tr0={p[4]:.2g} tc={p[5]:.0f}")
