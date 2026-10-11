import pickle,numpy as np
runs=[("out/frz_0.2_0.005_0.2_0.005_1.pkl",0.005,0.005),("out/frz_0.2_0.005_0.2_0.0025_3.pkl",0.005,0.0025),("out/frz_0.2_0.01_0.25_0.005_2.pkl",0.01,0.005),("out/frz_0.2_0.02_0.35_0.005_4.pkl",0.02,0.005)]
for fn,d,sig in runs:
    h=pickle.load(open(fn,'rb')); t,rho,f,G,dd=h.T
    B=pickle.load(open(f"out/bgrid_{d}.pkl","rb")); beta=lambda x: np.interp(x,B[:,0],B[:,2]); uu=lambda x: np.exp(np.interp(x,B[:,0],np.log(B[:,1])))
    fp=f[0]; Gp=0.0; out=[]; T=t[-1]; tt=0.0; dt=10.0; marks=set(np.unique(np.geomspace(1,len(t)-1,9).astype(int)))
    rec={}
    while tt<T:
        fp+=dt*Gp*beta(fp); Gp+=dt*sig**2*uu(fp); tt+=dt; rec[int(round(tt))]=(fp,Gp)
    print(fn.split('/')[1]," ".join(f"t={int(t[i])}: f {f[i]:.3f}/{rec.get(int(t[i]),(np.nan,0))[0]:.3f} sd {np.sqrt(G[i]):.4f}/{np.sqrt(rec.get(int(t[i]),(0,np.nan))[1]):.4f}" for i in sorted(marks)))
