import numpy as np, xanal
from functools import lru_cache
from qsd import qsd
from spitz import pi_exact
from xtheory import ingredients
fg=np.linspace(0.04,0.9,30)
rat=np.array([pi_exact(f,nexact=200)/ingredients(f)[5] for f in fg])
np.save('spitz_ratio.npy',np.c_[fg,rat])
@lru_cache(None)
def ing(f):
    q=qsd(f); return q['g'],q['s2'],np.interp(f,fg,rat)*q['pi'],q['Tr']
xanal.ing=ing
print(np.round(rat,3))
for mu in [0,1e-4,3e-4,1e-3,3e-3,0.005,0.01,0.02,0.03,0.035,0.04,0.045]:
    print(mu,[round(x,4) for x in xanal.fstar(mu)],flush=True)
