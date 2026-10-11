import numpy as np, pickle
from rob2 import ess
from multiprocessing import Pool
base=dict(m=0.2,var=1.0,d=0.01,wmin=0.1,wcap=2.0)
cases=[dict()]
for k,vals in dict(wmin=[0.02,0.05,0.2,0.4],wcap=[1.0,4.0,8.0],d=[0.0,0.005,0.02,0.04],var=[0.25,0.5,2.0],m=[0.05,0.1,0.3]).items():
    for v in vals: cases.append({k:v})
def job(c):
    kw=dict(base); kw.update(c); kw['e']=0.0
    try: return c, ess('lottery',kw,0.1,1.0)
    except Exception as ex: return c, repr(ex)
if __name__=='__main__':
    with Pool(2) as p: res=p.map(job,cases)
    pickle.dump(res,open('lotscan.pkl','wb'))
    for c,f in res: print(c,f)
