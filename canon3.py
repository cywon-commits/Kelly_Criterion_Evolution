import pickle,numpy as np
runs=[("out/frz_0.2_0.005_0.2_0.005_1.pkl",0.005,0.005),("out/frz_0.2_0.005_0.2_0.0025_3.pkl",0.005,0.0025),("out/frz_0.2_0.01_0.25_0.005_2.pkl",0.01,0.005),("out/frz_0.2_0.02_0.35_0.005_4.pkl",0.02,0.005)]
res={}
for fn,d,sig in runs:
    h=pickle.load(open(fn,'rb')); t,rho,f,G,dd=h.T
    C=pickle.load(open(f"out/cgrid_{d}.pkl","rb"))
    beta=lambda x: np.interp(x,C[:,0],C[:,2]); uu=lambda x: np.exp(np.interp(x,C[:,0],np.log(C[:,1]))); cc=lambda x: np.interp(x,C[:,0],C[:,4])
    fp=f[0]; Gp=0.0; tt=0.0; dt=10.0; rec={}
    while tt<t[-1]:
        fp+=dt*Gp*beta(fp); Gp=max(Gp+dt*(sig**2*uu(fp)+Gp**2*cc(fp)),0); tt+=dt; rec[int(round(tt))]=(fp,Gp)
    marks=sorted(set(np.unique(np.geomspace(1,len(t)-1,9).astype(int))))
    print(fn.split('/')[1]," ".join(f"t={int(t[i])}: f {f[i]:.3f}/{rec[int(t[i])][0]:.3f} sd {np.sqrt(G[i]):.4f}/{np.sqrt(rec[int(t[i])][1]):.4f}" for i in marks))
    res[fn]=(t,f,np.sqrt(G),np.array([rec[int(x)][0] if int(x) in rec else f[0] for x in t]),np.array([np.sqrt(rec[int(x)][1]) if int(x) in rec else 0 for x in t]))
pickle.dump(res,open("out/canon3.pkl","wb"))
