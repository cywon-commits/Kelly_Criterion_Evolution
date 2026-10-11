import sys; from wp1s import *
p=float(sys.argv[1]); d=float(sys.argv[2]); e=float(sys.argv[3]); M=int(sys.argv[4]); f0=float(sys.argv[5]) if len(sys.argv)>5 else 2*p-1
f,v=ess(f0,p=p,d=d,e=e,M=M)
print(f"p={p} d={d} e={e} M={M}: f/Kelly={(f/(2*p-1)) if f else float('nan'):.3f} v={v if v else float('nan'):.3e}",flush=True)
