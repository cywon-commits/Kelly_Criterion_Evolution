import sys, numpy as np, pickle
from hub1 import run
e=float(sys.argv[1]); sg=float(sys.argv[2]); T=int(sys.argv[3]); seed=int(sys.argv[4])
fi=lambda rng,N: rng.uniform(0.05,0.75,N)
(fh,fr),s=run(2,False,True,fi,e,fh0=0.3,fr0=0.3,T=T,sigma=sg,seed=seed)
pickle.dump(dict(fh=fh,fr=fr,s=s),open(f'hub1_{e}_{sg}_{seed}.pkl','wb'))
allf=np.r_[fh,fr]
print(e,sg,seed,'mean hub %.3f rim %.3f all %.3f sd %.4f'%(fh.mean(),fr.mean(),allf.mean(),allf.std()),'quantiles',np.round(np.quantile(allf,[0.05,0.25,0.5,0.75,0.95]),3),flush=True)
print('  hist',np.histogram(allf,bins=12,range=(0.15,0.75))[0])
