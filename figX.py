import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B='#2a78d6'; O='#eb6834'; G='#1baf7a'; INK='#222'; GR='#888'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'font.family':'DejaVu Sans'})
fK=0.330; Tr=1.32e4
al=[(6.26e-5,0.116),(4.92e-4,0.140),(1.23e-3,0.163),(3.16e-3,0.199),(6.91e-3,0.250),(1.43e-2,0.329),(2.36e-2,0.423),(3.35e-2,0.530),(3.95e-2,0.583),(4.15e-2,0.635),(4.24e-2,0.635)]
ee=[(1e-4,0.119),(3e-4,0.132),(1e-3,0.157),(3e-3,0.196),(5e-3,0.224),(1e-2,0.286),(2e-2,0.390),(3e-2,0.482),(3.5e-2,0.569),(4e-2,0.613),(4.2e-2,0.652)]
lot=[(0.0424,0.613)]
fig,ax=plt.subplots(figsize=(5.6,3.6))
X=lambda a: np.array([p[0] for p in a])*Tr; Y=lambda a: np.array([p[1] for p in a])/fK
ax.plot(X(ee),Y(ee),'s-',color=O,ms=6,lw=1.6,label='background mortality $e$ (birth regulation)')
ax.plot(X(al),Y(al),'o',color=B,ms=6,mfc='white',mew=1.6,label='density-dependent mortality ($\\alpha$ scan, $e=0$)')
ax.plot(X(lot),Y(lot),'D',color=G,ms=8,mec='white',mew=1.2,label='pure displacement (lottery)',zorder=5)
ax.plot([0.03],[0.108/fK],'o',color=B,ms=6,mfc='white',mew=1.6,clip_on=False)
ax.set_xscale('log'); ax.set_xlim(0.03,1200); ax.set_ylim(0,2.3)
ax.axhline(1,color=GR,ls='--',lw=1); ax.text(0.04,1.04,'Kelly',color=GR,fontsize=8)
ax.axhline(0.108/fK,color=GR,ls=':',lw=1); ax.text(0.04,0.108/fK+0.05,'$2\\delta/m$ (law I)',color=GR,fontsize=8)
ax.axvline(1,color=GR,lw=0.8,ls='-.'); ax.text(1.12,0.62,'onset\n$X\\approx1$',fontsize=8,color=INK)
xe=0.0424*Tr; ax.axvline(xe,color=INK,lw=1); ax.text(xe*0.92,0.12,'extinction edge\n$\\mu_{\\rm ext}=e_c^{\\rm evo}$',ha='right',fontsize=8,color=INK)
ax.annotate('waiting world',(0.03,0.108/fK),xytext=(0.05,0.62),fontsize=8.5,color=B,arrowprops=dict(arrowstyle='-',color=B,lw=0.8))
ax.text(xe*0.8,2.12,'displacement world',ha='right',fontsize=8.5,color=G)
ax.set_xlabel('$X=\\mu_{\\rm ext}\\,T_r$   (extrinsic mortality per ruin time)')
ax.set_ylabel('evolved betting fraction $f^*/f_K$')
ax.legend(fontsize=7.2,loc='upper left',bbox_to_anchor=(0.0,0.82))
ax.text(0.03,-0.035,'0',transform=ax.get_xaxis_transform(),va='top',fontsize=8,ha='center',color='#444')
fig.tight_layout()
for ext in ['pdf','png']: fig.savefig(f'/mnt/user-data/outputs/figs/fig_crossing.{ext}',dpi=220,bbox_inches='tight')
