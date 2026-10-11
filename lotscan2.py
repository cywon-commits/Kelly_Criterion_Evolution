import numpy as np, pickle
from robust import env_res
from multiprocessing import Pool
KW=dict(M=500,K=150)
def ess(kw,lo,hi,n=25):
    fs=np.linspace(lo,hi,n); E=np.array([np.log10(max(env_res(f,'lottery',**kw,**KW) or 1e-14,1e-14)) for f in fs])
    i=int(np.argmax(E)); j=min(max(i,2),n-3); c=np.polyfit(fs[j-2:j+3],E[j-2:j+3],2)
    fstar=-c[1]/(2*c[0]) if c[0]<0 else fs[i]
    return float(np.clip(fstar,lo,hi)), fs, E
def job(a):
    m,V,wmin=a; kw=dict(m=m,var=V,d=0.01,wmin=wmin,wcap=2.0,e=0.0)
    r=m/V; lo,hi=max(0.02,0.8*r),min(1.0,5*r)
    f1,fs,E=ess(kw,lo,hi)
    w=(hi-lo)/6; f2,fs2,E2=ess(kw,max(0.01,f1-w),min(1.0,f1+w),15)
    return a,(f2,fs,E,fs2,E2)
if __name__=='__main__':
    cases=[(m,V,w) for (m,V) in [(0.2,1),(0.1,0.5),(0.4,2),(0.1,1),(0.2,2),(0.05,0.5)] for w in [0.4,0.2,0.1,0.05,0.02]]
    with Pool(2) as p: res=dict(p.map(job,cases))
    pickle.dump(res,open('lotscan2.pkl','wb'))
    for a in cases: print(a,'b=%.2f'%np.log(1/a[2]), 'f*=%.3f'%res[a][0], 'F=%.2f'%(res[a][0]*a[1]/a[0]))
