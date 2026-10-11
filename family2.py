import pickle
from family import run
from multiprocessing import Pool
if __name__=='__main__':
    fams=[('Q0.7',0.2,1.0,0.02/0.7,0.1),('Q1.5',0.2,1.0,0.02/1.5,0.1),('Q3',0.2,1.0,0.02/3,0.1),('Q2.5',0.2,1.0,0.02/2.5,0.1)]
    with Pool(2) as p: res=p.map(run,fams)
    pickle.dump(res,open('family2.pkl','wb'))
