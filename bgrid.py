import numpy as np, sys, pickle
from chainT import rates
d=float(sys.argv[1]); lo,hi=float(sys.argv[2]),float(sys.argv[3])
fs=np.linspace(lo,hi,int(sys.argv[4])); out=[]
for f in fs:
    try: r=rates(f,d=d); out.append((f,r['u'],r['beta'],r['v']))
    except Exception: pass
pickle.dump(np.array(out),open(f"out/bgrid_{d}.pkl","wb")); print(d,"done",len(out))
