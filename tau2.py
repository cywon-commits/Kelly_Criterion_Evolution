import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, sys
from tau import build, tau
kw=dict(M=1200,K=150)
def pieces(f):
    x,A,hc,steps=build(f,ruin=False,**kw); M=len(x)
    # stationary distribution of no-ruin walk (reflect at bottom): left eigvec
    w,V=spl.eigs(A.T.tocsc(),k=1,sigma=1.0); pi=np.abs(np.real(V[:,0])); pi/=pi.sum()
    pcap=pi@hc
    # mean first passage time to cap (first arrival) from x0 : solve (I - A_nocap) T = 1 ; A without cap column
    A2=A.tolil(); A2[:,M-1]=0; A2=A2.tocsr()
    Tc=spl.spsolve((sp.eye(M,format='csr')-A2).tocsc(),np.ones(M)); tc=np.interp(np.log(1.0),x,Tc)
    # ruin rate at e=0, v=0: mean time to ruin
    xr,Ar,hcr,_=build(f,ruin=True,**kw); Tr=spl.spsolve((sp.eye(M,format='csr')-Ar).tocsc(),np.ones(M)); tr=np.interp(0.0,xr,Tr)
    return pcap,tc,tr
fs=np.linspace(0.05,0.45,41); P=[pieces(f) for f in fs]
import pickle; pickle.dump((fs,P),open('out/pieces.pkl','wb'))
for f,(a,b,c) in list(zip(fs,P))[::4]: print(f"f={f:.2f} pcap={a:.4f} t_climb={b:.1f} T_ruin={c:.3g}")
