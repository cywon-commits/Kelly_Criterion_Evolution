import pickle, numpy as np
C=pickle.load(open('out/cgrid_0.005.pkl','rb'))
beta=lambda x: np.interp(x,C[:,0],C[:,2]); uu=lambda x: np.exp(np.interp(x,C[:,0],np.log(C[:,1]))); cc=lambda x: np.interp(x,C[:,0],C[:,4])
def traj(N,T,sig=0.005,f0=0.2,drift=True,dt=10.0):
    f=f0; G=0.0; t=0.0; out={}
    marks=[1e4,3e4,1e5,2e5,4e5,1e6,1e7]
    while t<T:
        f+=dt*G*beta(f); G=max(G+dt*(sig**2*uu(f)+G**2*cc(f)-(2*uu(f)/N*G if drift else 0)),0); t+=dt
        for mk in marks:
            if abs(t-mk)<dt/2: out[mk]=(f,np.sqrt(G))
    return out
if __name__=="__main__":
    for N in [256,1024,4096,16384]:
        a=traj(N,4.1e5); b=traj(N,4.1e5,drift=False)
        print(N,' '.join(f"{int(k)}:{v[0]:.3f}(sd{v[1]:.4f})" for k,v in a.items()),'| no-drift 4e5:',f"{b[4e5][0]:.3f}")
