from kelly import run
import numpy as np
for xi in [1,4,16,64]:
  row=[]
  for fx in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6]:
    h=run(xi=xi,fixed=fx,T=3000)
    row.append(f"{fx}:{np.mean([r[1] for r in h[-20:]]):.2f}" if h[-1][1]>0 else f"{fx}:X@{h[-1][0]}")
  print("xi",xi," ".join(row))
