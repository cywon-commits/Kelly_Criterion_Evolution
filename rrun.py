import sys, json, numpy as np
from robust import ess
name=sys.argv[1]; variant=sys.argv[2]; kw=json.loads(sys.argv[3])
fs,E,f1=ess(variant,0.02,0.6,30,**kw)
fs2,E2,f2=ess(variant,max(f1-0.025,0.01),f1+0.025,11,**kw)
envname='mu_d' if variant=='lottery' else 'v_r'
ev=E2[np.nanargmax(E2)] if variant=='lottery' else E2[np.nanargmin(E2)]
print(f"{name:28s} ESS f*={f2:.3f}  ({envname} at ESS = {ev:.2e})",flush=True)
