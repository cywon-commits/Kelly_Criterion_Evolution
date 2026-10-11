import numpy as np, sys
from wp1s import P0, vres
p,d,e=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); frs=[float(x) for x in sys.argv[4].split(',')]; fis=[float(x) for x in sys.argv[5].split(',')]
kw=dict(p=p,d=d,e=e,M=3000); K=2*p-1
for fr in frs:
    v=vres(fr,**kw)
    print(f"p={p} d={d} e={e} resident {fr/K:.3f}K (v={v:.3g}): "+" ".join(f"{fi/K:.2f}K:{2*P0(fi,v,**kw)-1:+.4f}" for fi in fis),flush=True)
