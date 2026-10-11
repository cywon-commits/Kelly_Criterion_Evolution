import pickle,numpy as np,glob
from ntheory import beta,uu,cc
def sim_stats(L):
    fs=sorted(glob.glob(f'out/nsim_L{L}_s*.pkl')) if L!=64 else ['out/frz_0.2_0.005_0.2_0.005_11.pkl','out/frz_0.2_0.005_0.2_0.005_1.pkl']
    out={}
    for mk in [3e4,1e5,2e5,3.99e5]:
        vals=[]
        for fn in fs:
            h=pickle.load(open(fn,'rb')); t,rho,f,G,dd=h.T
            if mk<=t[-1]+1: i=min(np.searchsorted(t,mk),len(t)-1); j=slice(max(i-40,0),i+1); vals.append((np.nanmean(f[j]),np.sqrt(np.nanmean(G[j]))))
        if vals: out[mk]=np.mean(vals,axis=0)
    return out
def theo(N,k,T=4e5,dt=10.0):
    f=0.2;G=0.0;t=0.0;out={}
    while t<T:
        f+=dt*G*beta(f); G=max(G+dt*(0.005**2*uu(f)+G**2*cc(f)-k*2*uu(f)/N*G),0); t+=dt
        for mk in [3e4,1e5,2e5,3.99e5]:
            if abs(t-mk)<dt/2: out[mk]=(f,np.sqrt(G))
    return out
S={L:sim_stats(L) for L in [16,32,64,128]}
for k in [1,4,10,20]:
    line=[]
    for L in [16,32,64,128]:
        N=L*L; th=theo(N,k)
        line.append(f"N={N}: "+" ".join(f"{int(mk)}:{S[L][mk][0]:.3f}/{th[mk][0]:.3f}" for mk in S[L]))
    print(f"k={k} | "+" || ".join(line))
