import numpy as np, pickle
from functools import lru_cache
from sitechain import site
from pairapp import solve as pair
from chainT import solve as chain
KH=24
@lru_cache(None)
def env(fr,e):
    vH,vR=0.5,0.1
    for k in range(40):
        sH=site(fr,vH,e); sR=site(fr,vR,e); r=pair(sH['d'],sR['d'],sH['pi'],sR['pi'])
        vH=0.5*vH+0.5*r['v_hub']; vR=0.5*vR+0.5*r['v_rim']
    q=sH['d']*r['rho_hub']/(KH*r['rho_rim']*sR['pi']*vR)   # fraction of rim births that go to a hub
    return vH,vR,min(q,1.0)
@lru_cache(None)
def P(f,v,e): return chain(f,v,e=e,M=800,K=150)[0]
def R(fm,fr,e):
    vH,vR,q=env(fr,e); PH=P(fm,vH,e); PR=P(fm,vR,e)
    K=np.array([[PH,PH],[PR*q,PR*(2-q)]])      # rows: start at H / at R ; cols: new starts at H, R
    return max(abs(np.linalg.eigvals(K)))
if __name__=='__main__':
    import sys
    e=float(sys.argv[1]); fs=np.round(np.linspace(float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])),4)
    S=np.array([[np.log(R(fm,fr,e)/R(fr,fr,e)) for fm in fs] for fr in fs])
    for fr in fs: print('res',fr,'env',[round(x,4) for x in env(fr,e)],'R(res,res)=%.4f'%R(fr,fr,e),flush=True)
    pickle.dump(dict(fs=fs,S=S),open(f'pip_{e}.pkl','wb'))
    # singular point: where gradient changes sign
    g=[(S[i,min(i+1,len(fs)-1)]-S[i,max(i-1,0)]) for i in range(len(fs))]
    print('gradient',np.round(g,5))
