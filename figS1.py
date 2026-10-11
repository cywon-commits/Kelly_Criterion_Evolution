import numpy as np, pickle, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']; INK='#222'; GREY='#888'
plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#555','xtick.color':'#444','ytick.color':'#444','lines.linewidth':1.6,'legend.frameon':False})
def lab(ax,s): ax.text(-0.16,1.05,s,transform=ax.transAxes,fontsize=11,fontweight='bold',color=INK)
fig,axs=plt.subplots(2,2,figsize=(7.4,6.0))
# (a) correction decomposition
ax=axs[0,0]; st=['$2\\delta/m$','exact\nmoments','+pre-\nfactor','+Lund-\nberg','+over-\nshoot','chain\nESS','simul-\nation']; v=[0.100,0.111,0.102,0.105,0.107,0.109,0.110]
ax.plot(range(len(v)),v,'-',color=GREY,lw=1,zorder=1)
for i,y in enumerate(v): ax.plot(i,y,'o' if i<5 else ('D' if i==5 else 's'),ms=7,color=C[0] if i<5 else (C[2] if i==5 else C[1]),zorder=2); ax.text(i,y+0.0015,f'{y:.3f}',ha='center',fontsize=7,color=INK)
ax.set_xticks(range(len(v))); ax.set_xticklabels(st,fontsize=7); ax.set_ylim(0.096,0.116); ax.set_ylabel('$f^*$'); ax.set_title('Corrections to $2\\delta/m$ (m=0.2, Var R=1, δ=0.01)',fontsize=8.5,loc='left'); lab(ax,'a')
# (b) binomial staircase vs continuous
ax=axs[0,1]; B={0:[0.45],2:[0.675],5:[0.76],10:[1.04,1.06,1.13],15:[1.18,1.29,1.38],20:[1.18,1.30,1.38,1.66],25:[1.38,1.56,1.66,2.15],30:[1.58,1.68,2.15]}
first=True
for e,ls in B.items():
    for y in ls: ax.plot(e,y,'_',ms=12,mew=2,color=C[1],label='binomial: local ESSs' if first else None); first=False
ce=[0,5,10,20]; cv=[0.109/0.33,0.225/0.33,0.285/0.33,0.395/0.33]
ax.plot(ce,cv,'o-',color=C[0],ms=6,label='lognormal: unique ESS')
ax.axhline(1,color=GREY,ls='--',lw=1); ax.set_xlabel('background mortality $e$ (×10$^{-3}$)'); ax.set_ylabel('$f^*/f_K$'); ax.legend(fontsize=7,loc='upper left')
ax.set_title('Binomial returns give a staircase of local ESSs',fontsize=8.5,loc='left'); ax.text(0.98,0.03,'binomial: p=0.6, $f_K$=0.2\nlognormal: m=0.2, $f_K$=0.33',transform=ax.transAxes,ha='right',fontsize=6.5,color=GREY); lab(ax,'b')
# (c) robustness dot plot
ax=axs[1,0]; rows=['reference','split 30:70','threshold 2 / cap 4','lottery (mortality reg.)']
vals=[[0.110,0.285,0.205,0.110],[0.110,0.285,0.210,0.110],[0.105,0.220,0.190,np.nan],[0.61,0.61,0.63,0.99]]
cols=['$e$=0','$e$=0.01','δ=0.02','Var R=0.25']
for j in range(4):
    for i in range(4):
        if not np.isnan(vals[i][j]): ax.plot(vals[i][j],3-i+(j-1.5)*0.14,'o',ms=6,color=C[j],label=cols[j] if i==0 else None)
ax.set_ylim(-0.6,3.5); ax.set_yticks(range(4)); ax.set_yticklabels(rows[::-1],fontsize=7.5); ax.axvline(0.33,color=GREY,ls='--',lw=1); ax.text(0.34,-0.45,'Kelly (Var R=1)',fontsize=7,color=GREY)
ax.set_xlabel('ESS $f^*$'); ax.legend(fontsize=7,loc='center right',bbox_to_anchor=(1.0,0.55)); ax.set_title('Robustness: only the site of regulation matters',fontsize=8.5,loc='left'); lab(ax,'c')
# (d) N dependence
ax=axs[1,1]
def load(L):
    fs=sorted(glob.glob(f'out/nsim_L{L}_s*.pkl')) if L!=64 else ['out/frz_0.2_0.005_0.2_0.005_11.pkl']
    H=[pickle.load(open(f,'rb')) for f in fs]; n=min(len(h) for h in H); return H[0][:n,0],np.mean([h[:n,2] for h in H],axis=0)
for i,(L,nm) in enumerate([(16,'N=256'),(32,'N=1024'),(64,'N=4096'),(128,'N=16384')]):
    t,f=load(L); k=max(1,len(t)//400); ax.plot(t[1::k],f[1::k],color=C[i],lw=1.2,label=nm)
from ntheory import beta,uu,cc
for kf,ls,nm in [(1,':','theory N=256, k=1'),(12,'--','theory N=256, k=12')]:
    f=0.2;G=0.0;t=0.0;T=[];F=[]
    while t<4e5:
        f+=10*G*beta(f); G=max(G+10*(0.005**2*uu(f)+G**2*cc(f)-kf*2*uu(f)/256*G),0); t+=10; T.append(t); F.append(f)
    ax.plot(T,F,ls,color=INK,lw=1.2,label=nm)
ax.set_xscale('log'); ax.set_xlim(1e4,4e5); ax.set_ylim(0.11,0.205); ax.set_xlabel('generation $t$'); ax.set_ylabel('$\\bar f$'); ax.legend(fontsize=6.5)
ax.set_title('Population-size dependence of freezing (δ=0.005)',fontsize=8.5,loc='left'); lab(ax,'d')
fig.tight_layout(); fig.savefig('/mnt/user-data/outputs/figs/figS1_appendix.pdf',bbox_inches='tight'); fig.savefig('/mnt/user-data/outputs/figs/figS1_appendix.png',dpi=200,bbox_inches='tight'); print('ok')
