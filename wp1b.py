import sys; from wp1s import P0; from scipy.optimize import minimize_scalar
p,d,wmin,v=map(float,sys.argv[1:5])
r=minimize_scalar(lambda f:-P0(f,v,p=p,d=d,wmin=wmin,M=3000),bounds=(0.002,0.98),method='bounded',options={'xatol':1e-4})
print(f"p={p} d={d} wmin={wmin} v={v}: argmax f/Kelly={r.x/(2*p-1):.3f}",flush=True)
