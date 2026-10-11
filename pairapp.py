import numpy as np
from scipy.optimize import fsolve
K=24  # rims per hub; each rim: 1 hub + 2 rims
def rhs(y,dH,dR,pH,pR):
    xH,xR,a,b=y   # a=P(H1,R1) on HR edge, b=P(R1,R1) on RR edge
    # conditionals
    qR1_H1=a/xH; qR0_H1=1-qR1_H1
    qR1_H0=(xR-a)/(1-xH)
    qH1_R1=a/xR; qH0_R1=1-qH1_R1
    qH1_R0=(xH-a)/(1-xR)
    qR1_R1=b/xR; qR0_R1=1-qR1_R1
    qR1_R0=(xR-b)/(1-xR)
    # expected # empty nbrs of an occupied site, given a specified neighbour is empty
    EH=1+(K-1)*qR0_H1                 # hub occupied, one rim empty
    ER_rimempty=1+qH0_R1+qR0_R1       # rim occupied, one rim nbr empty (other: hub + 1 rim)
    ER_hubempty=1+2*qR0_R1            # rim occupied, its hub empty
    fH=pH/EH; fRr=pR/ER_rimempty; fRh=pR/ER_hubempty   # per-link filling rates
    # filling rate of empty rim given neighbour states (closure)
    FR=qH1_R0*fH+2*qR1_R0*fRr
    FH=K*qR1_H0*fRh
    dxH=-dH*xH+(1-xH)*FH
    dxR=-dR*xR+(1-xR)*FR
    # HR edge: (H0,R1)-> H filled: via this rim fRh + other K-1 rims (K-1)qR1_H0 fRh ; (H1,R0)-> R filled: via this hub fH + 2 rims qR1_R0 fRr
    pH0R1=xR-a; pH1R0=xH-a
    da=-(dH+dR)*a+pH0R1*(fRh+(K-1)*qR1_H0*fRh)+pH1R0*(fH+2*qR1_R0*fRr)
    # RR edge: (R0,R1): empty one filled via this nbr fRr + its hub qH1_R0 fH + its other rim qR1_R0 fRr
    pR0R1=xR-b
    db=-2*dR*b+2*pR0R1*(fRr+qH1_R0*fH+qR1_R0*fRr)
    return [dxH,dxR,da,db]
def solve(dH,dR,pH,pR,y0=None):
    if y0 is None: y0=[0.99,0.8,0.8,0.7]
    # integrate to steady state
    y=np.array(y0,float); dt=0.5
    for it in range(400000):
        dy=np.array(rhs(y,dH,dR,pH,pR)); y=y+dt*dy
        y=np.clip(y,1e-9,1-1e-9)
        if np.abs(dy).max()<1e-12: break
    xH,xR,a,b=y
    vH=1-(a/xH)**K; vR=1-(a/xR)*(b/xR)**2
    vH_mf=1-xR**K; vR_mf=1-xH*xR**2
    return dict(rho_hub=xH,rho_rim=xR,v_hub=vH,v_rim=vR,v_hub_mf=vH_mf,v_rim_mf=vR_mf)
if __name__=='__main__':
    print(solve(0.0143,0.0117,0.0347,0.0745))
    print(solve(0.0017,0.0001,0.1044,0.0765,[0.999,0.999,0.998,0.998]))
