import numpy as np
from scipy.stats import norm
from cont import lnparams
mu,s=lnparams(0.2,1.0)
def pi_exact(f,d=0.01,N=3000,nexact=400):
    # step xi=ln(1+fR)+ln(1-d); Spitzer: P(W=0)=exp(-sum P(S_n<0)/n)  (W=distance below cap, Lindley)
    u=norm.ppf((np.arange(20000)+0.5)/20000); xi=np.log1p(f*(np.exp(mu+s*u)-1))+np.log(1-d)
    g,v=xi.mean(),xi.var()
    L=4*(abs(g)*nexact+6*np.sqrt(v*nexact)+xi.max()-xi.min()); dx=L/2**18
    grid=np.zeros(2**18); idx=np.round(xi/dx).astype(int)%2**18; np.add.at(grid,idx,1/len(xi))
    F=np.fft.fft(grid); acc=0; Fn=np.ones_like(F)
    k=np.fft.fftfreq(2**18)*2**18; neg=(k<0)
    for n in range(1,nexact+1):
        Fn=Fn*F; p=np.fft.ifft(Fn).real; acc+=p[neg].sum()/n
    n=np.arange(nexact+1,N+1); acc+=np.sum(norm.cdf(-np.sqrt(n)*g/np.sqrt(v))/n)
    return np.exp(-acc)
if __name__=='__main__':
    from xhybrid import chain_ing
    from xtheory import ingredients
    for f in [0.1,0.2,0.3,0.4,0.5]:
        print(f,'gauss %.3f exact %.3f chain %.3f'%(ingredients(f)[5],pi_exact(f),chain_ing(f)[0]),flush=True)
