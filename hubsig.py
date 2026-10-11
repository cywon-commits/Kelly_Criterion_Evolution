import sys
from hubp import run
e=float(sys.argv[1]); sg=float(sys.argv[2]); fh=float(sys.argv[3]); T=int(sys.argv[4])
s=run(2,e,fh0=fh,fr0=0.39,T=T,sigma=sg,seed=3)
print(e,sg,fh,{k:round(float(v),3) for k,v in s.items()},flush=True)
