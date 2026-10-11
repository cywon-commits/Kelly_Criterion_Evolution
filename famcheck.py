import pickle,numpy as np
from rob2 import ess
from multiprocessing import Pool
R={r['name']:r for r in pickle.load(open('family.pkl','rb'))}
def job(a):
    name,u=a; r=R[name]; mu=u*r['muc']
    kw=dict(m=r['m'],var=r['V'],d=r['d'],wmin=r['wmin'],wcap=2.0,e=mu)
    f=ess('vacancy',kw,0.03,0.95)
    rows=np.array([(m_,f_) for m_,f_,v in r['rows'] if np.isfinite(f_)])
    return name,u,f,float(np.interp(mu,rows[:,0],rows[:,1]))
if __name__=='__main__':
    jobs=[('Q4',0.5),('Q1',0.5),('b1.6',0.5),('b3.0',0.5),('Q4',0.1),('Q1',0.1)]
    with Pool(2) as p:
        for name,u,f,th in p.imap(job,jobs): print(name,u,'population ESS %.3f renewal %.3f'%(f,th),flush=True)
