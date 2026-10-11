import numpy as np, pickle, glob, scipy.sparse as sp, scipy.sparse.linalg as spl
from tau import build
for fn in sorted(glob.glob('out/univ_*.pkl')):
    D=pickle.load(open(fn,'rb')); kw=dict(m=D['m'],var=D['V'],d=D['d'],M=1000,K=120); row=[]
    for X,e,f in D['out']:
        x,A,hc,_=build(f,**kw); Tr=spl.spsolve((sp.eye(len(x),format='csr')-A).tocsc(),np.ones(len(x))); r=1/np.interp(0.0,x,Tr)
        row.append(f"X={X:g}:{r/e:.3f}")
    print(f"m={D['m']} V={D['V']} d={D['d']} ruin/e:",' '.join(row),flush=True)
