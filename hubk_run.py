import sys
from hubk import run
e=float(sys.argv[1]); fh=float(sys.argv[2]); T=int(sys.argv[3])
s=run(2,True,e,fh0=fh,fr0=0.39 if e>0.015 else 0.29,T=T,sigma=0.003,seed=5)
print('NOKIN',e,fh,{k:round(float(v),3) for k,v in s.items()},flush=True)
