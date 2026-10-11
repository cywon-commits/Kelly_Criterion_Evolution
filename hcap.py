import numpy as np
from robust import kernel, interp_row
from family import Fam
F=Fam(0.2,1.0,0.01,0.1)
def exact(f,v,mu,M=500,K=150):
    x,T=kernel(f,0.2,1.0,0.01,0.1,2.0,M,K); cap=np.zeros(M); cap[-1]=1
    s=(1-mu); A=s*T*(1-v*cap)[None,:]; B=s*T*(v*cap)[None,:]
    Nrep=np.linalg.solve(np.eye(M)-A,B.sum(1))   # prob of reproducing before death, from each state
    st=interp_row(x,np.log(1.0))
    return st@Nrep, Nrep[-1]
def Hrace(f,v,mu):
    pi,Tr,*_=F.ing(f); h=mu+1/Tr; return v+(1-v)*pi*v/(pi*v+h)
for f in [0.1,0.3,0.6]:
    for mu in [0.0,0.01,0.03]:
        for v in [0.001,0.1,1.0]:
            P,Hc=exact(f,v,mu); Pi=F.Pi(f,mu); Harr=v+(1-v)*Hc
            print('f=%.1f mu=%.3f v=%.3f  P=%.4f  Pi*Harr=%.4f  Harr=%.4f  race=%.4f'%(f,mu,v,P,Pi*Harr,Harr,Hrace(f,v,mu)))

from scipy.stats import norm
from cont import lnparams
def q0(f,m=0.2,V=1.0,d=0.01):
    mu_,s_=lnparams(m,V); thr=np.log(1+(1/(1-d)-1)/f)   # need (1+fR)(1-d)>=1 -> R>=thr_R ; e^Z >= 1+thr_R
    return 1-norm.cdf((np.log(1+(1/(1-d)-1)/f)-mu_)/s_)
def H2(f,v,mu):
    pi,Tr,*_=F.ing(f); h=mu+1/Tr; Hr=pi*v/(pi*v+h); q=q0(f)
    return (v+(1-v)*(1-mu)*(1-q)*Hr)/(1-(1-v)*(1-mu)*q)
print('--- improved')
err1=[];err2=[]
for f in [0.1,0.2,0.3,0.45,0.6]:
    for mu in [0.0,0.001,0.01,0.03]:
        for v in [0.001,0.01,0.1,0.5]:
            P,Hc=exact(f,v,mu); Harr=v+(1-v)*Hc
            err1.append(Hrace(f,v,mu)/Harr-1); err2.append(H2(f,v,mu)/Harr-1)
err1=np.abs(err1); err2=np.abs(err2)
print('race: median %.3f max %.3f ; two-state: median %.3f max %.3f'%(np.median(err1),err1.max(),np.median(err2),err2.max()))
