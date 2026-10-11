from kelly import run
import numpy as np, sys, json
kw=json.loads(sys.argv[1]); T=int(sys.argv[2]); p=kw.get('p',0.6); fk=2*p-1
out=[]
for s in [11,12]:
    h=np.array(run(T=T,f0=fk,seed=s,**kw))
    if h[-1,1]==0: out.append(f"EXTINCT@{int(h[-1,0])}"); continue
    q=h[-len(h)//5:]
    out.append(f"{np.nanmean(q[:,2])/fk:.2f}(rho{np.mean(q[:,1]):.2f})")
print(kw,T,out,flush=True)
