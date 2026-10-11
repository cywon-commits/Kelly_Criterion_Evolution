import numpy as np, pickle, sys
from cont import run
tag=sys.argv[1]; e=float(sys.argv[2]); lot=sys.argv[3]=='1'; sg=float(sys.argv[4]); T=int(sys.argv[5])
rng=np.random.default_rng(7); f0=rng.uniform(0.05,0.75,4096)
import cont
# record full distribution snapshots
snaps={}
h=run(L=64,e=e,T=T,sigma=sg,f_init=f0,lottery=lot,seed=11)
pickle.dump(h,open(f'dist_{tag}.pkl','wb'))
h=np.array(h); print(tag, 'final t',h[-1,0],'mean',round(h[-1,2],3),'sd',round(np.sqrt(h[-1,3]),4), flush=True)
for t in [0,1000,5000,20000,50000,T-50]:
    i=np.argmin(abs(h[:,0]-t)); print('  t=%d mean=%.3f sd=%.4f'%(h[i,0],h[i,2],np.sqrt(h[i,3])))
