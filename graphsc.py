import numpy as np, pickle
from sitechain import site
from pairapp import solve
def fstar(v,e):
    R=pickle.load(open(f'out/fv_{e}.pkl','rb')); return float(np.interp(np.log10(v),R[:,0],R[:,1]))
def selfc(e,fH=None,fR=None,it=40,br=True):
    vH,vR=0.5,0.1; y0=None
    for k in range(it):
        if br: fH,fR=fstar(vH,e),fstar(vR,e)
        sH=site(fH,vH,e); sR=site(fR,vR,e)
        r=solve(sH['d'],sR['d'],sH['pi'],sR['pi'])
        vH=0.5*vH+0.5*r['v_hub']; vR=0.5*vR+0.5*r['v_rim']
    return dict(fH=fH,fR=fR,vH=vH,vR=vR,ratio=fH/fR,rhoH=r['rho_hub'],rhoR=r['rho_rim'])
if __name__=='__main__':
    print('given sim f, e=0.01:',selfc(0.01,0.354,0.288,br=False))
    print('full, e=0.01:',selfc(0.01))
