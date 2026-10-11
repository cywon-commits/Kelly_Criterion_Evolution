import pickle,numpy as np
runs=[("out/frz_0.2_0.005_0.2_0.005_1.pkl",0.005),("out/frz_0.2_0.005_0.2_0.0025_3.pkl",0.005),("out/frz_0.2_0.01_0.25_0.005_2.pkl",0.01),("out/frz_0.2_0.02_0.35_0.005_4.pkl",0.02)]
for fn,d in runs:
    h=pickle.load(open(fn,'rb')); t,rho,f,G,dd=h.T
    B=pickle.load(open(f"out/bgrid_{d}.pkl","rb")); beta=lambda x: np.interp(x,B[:,0],B[:,2])
    # integrate df/dt = G(t)*beta(f) with measured G, dt=50
    fp=np.empty_like(f); fp[0]=f[0]
    for i in range(1,len(t)):
        fp[i]=fp[i-1]+(t[i]-t[i-1])*G[i-1]*beta(fp[i-1])
    idx=np.unique(np.geomspace(1,len(t)-1,9).astype(int))
    print(fn.split('/')[1], " ".join(f"t={int(t[i])}:{f[i]:.3f}/{fp[i]:.3f}" for i in idx))
