import numpy as np, pickle, sys
from tau import fstar_tau
fs,P=pickle.load(open('out/pieces.pkl','rb')); P=np.array(P); pc,tc,tr=P.T
ff=np.linspace(fs[0],fs[-1],2001)
PC=np.interp(ff,fs,pc); TC=np.exp(np.interp(ff,fs,np.log(tc))); TR=np.exp(np.interp(ff,fs,np.log(tr)))
es=[float(x) for x in sys.argv[1].split(',')]
for e in es:
    h=e+1/TR; F=PC*np.exp(-h*TC)/h; fa=ff[np.argmax(F)]
    fx=fstar_tau(e,lo=0.05,hi=0.45,M=1200,K=150)
    print(f"e={e:.0e}: exact={fx:.3f} approx={fa:.3f}  (ruin rate at f*: {1/np.interp(fx,ff,TR):.1e})",flush=True)
