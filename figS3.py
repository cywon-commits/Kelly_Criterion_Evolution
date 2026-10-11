import numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
B='#2a78d6'; O='#eb6834'; INK='#222'; GR='#888'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'font.family':'DejaVu Sans'})
fig,axs=plt.subplots(1,3,figsize=(9.6,3.0))
for k,(e,fsing,fh,fr) in enumerate([(0.01,0.295,0.331,0.284),(0.02,0.40,0.430,0.391)]):
    d=pickle.load(open(f'pip_{e}.pkl','rb')); fs,S=d['fs'],d['S']
    ax=axs[k]; ax.contourf(fs,fs,(S>1e-6).astype(float),levels=[-0.5,0.5,1.5],cmap=ListedColormap(['#f2f2f0','#9cc3ee']))
    ax.plot(fs,fs,color=INK,lw=0.8); ax.plot([fsing],[fsing],'o',color=INK,ms=6)
    ax.set_xlabel('resident $f$'); ax.set_ylabel('mutant $f$'); ax.set_title(f'$e={e}$: invasion (blue = mutant invades)',fontsize=8.5,loc='left')
    ax.text(-0.18,1.04,'ab'[k],transform=ax.transAxes,fontsize=11,fontweight='bold')
ax=axs[2]
for e,c,lbl in [(0.01,B,'$e=0.01$'),(0.02,O,'$e=0.02$')]:
    r=pickle.load(open(f'hub1_{e}_0.003_1.pkl','rb')); allf=np.r_[r['fh'],r['fr']]
    ax.hist(allf,bins=np.linspace(0.15,0.6,24),color=c,alpha=0.75,label=lbl+' (single gene, sim.)')
for x,c in [(0.295,B),(0.40,O)]: ax.axvline(x,color=c,ls='--',lw=1.2)
ax.set_ylim(0,960)
for x,c in [(0.284,B),(0.331,B),(0.391,O),(0.430,O)]: ax.plot([x],[925],'v',color=c,ms=6)
ax.set_xlabel('$f$ at $t=3\\times10^5$'); ax.set_ylabel('individuals'); ax.legend(fontsize=7,loc='center right')
ax.set_title('dashed: predicted singular $f$; ▼ rim / hub optima',fontsize=8,loc='left')
ax.text(-0.18,1.04,'c',transform=ax.transAxes,fontsize=11,fontweight='bold')
fig.tight_layout(); fig.savefig('/mnt/user-data/outputs/figs/figS3_single_gene.png',dpi=200,bbox_inches='tight'); fig.savefig('/mnt/user-data/outputs/figs/figS3_single_gene.pdf',bbox_inches='tight')
