from cont import run
import numpy as np, sys
mode=sys.argv[1]
if mode=='ramp':
    m,d,e0,e1,T,f0,seed=float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),int(sys.argv[6]),float(sys.argv[7]),int(sys.argv[8])
    h=np.array(run(m=m,delta=d,f0=f0,T=T,sigma=0.005,seed=seed,eramp=(e0,e1)))
    tx=h[-1,0]; ex=e0+(e1-e0)*tx/T
    ok=h[:,1]>0; last=h[ok][-40:]
    print(f"ramp m={m} seed={seed}: extinct at t={int(tx)} e={ex:.4f}; last f={np.nanmean(last[:,2]):.3f} rho={np.mean(last[:,1]):.3f}" if h[-1,1]==0 else f"ramp survived to e={e1}",flush=True)
else:
    m,d=float(sys.argv[2]),float(sys.argv[3])
    for fx,e in [(float(a),float(b)) for a,b in (x.split(':') for x in sys.argv[4].split(','))]:
        h=np.array(run(m=m,delta=d,fixed=fx,e=e,T=30000,seed=3))
        print(f"fixed m={m} f={fx} e={e}: {'EXTINCT t=%d'%h[-1,0] if h[-1,1]==0 else 'survive rho=%.3f'%np.mean(h[-20:,1])}",flush=True)
