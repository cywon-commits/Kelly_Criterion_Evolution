import numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']; INK='#222'; GREY='#888'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'font.family':'DejaVu Sans'})
OUT='/mnt/user-data/outputs/figs/'
res=pickle.load(open('lotscan2.pkl','rb'))
fig,axs=plt.subplots(1,2,figsize=(7.4,3.0))
ax=axs[0]; pairs=[(0.2,1),(0.1,0.5),(0.4,2),(0.1,1),(0.2,2),(0.05,0.5)]; mk='osD^vP'
for i,(m,V) in enumerate(pairs):
    pts=[(np.log(1/w),r[0]*V/m) for (mm,VV,w),r in res.items() if (mm,VV)==(m,V) and not (r[0]<0.11 and m<0.1)]
    pts.sort(); b,F=zip(*pts); ax.plot(b,F,mk[i]+'-',color=C[i],ms=5,lw=1,label=f'$m$={m:g}, Var$R$={V:g}')
ax.set_xlabel('ruin buffer $b=\\ln(w_{\\rm cap}/2w_{\\min})$'); ax.set_ylabel('$f^*\\,\\mathrm{Var}R/m$'); ax.set_ylim(0.8,4.6)
ax.legend(fontsize=6.5,loc='lower right'); ax.set_title('Law II: mortality-regulated ESS',fontsize=9,loc='left')
ax.text(-0.14,1.04,'a',transform=ax.transAxes,fontsize=11,fontweight='bold')
ax=axs[1]; es=np.array([0.005,0.01,0.02])
th_h=[0.260,0.331,0.430]; th_r=[0.223,0.284,0.391]
kin_h=[0.271,0.358,0.585]; kin_h_err=[0.003,0.012,0.045]; kin_r=[0.233,0.293,0.398]
nk_h=[0.253,0.311,0.40]; nk_r=[0.213,0.273,0.368]
ax.plot(es,th_h,'-',color=C[1],lw=1.4,label='theory, hub'); ax.plot(es,th_r,'-',color=C[0],lw=1.4,label='theory, rim')
ax.errorbar(es,kin_h,yerr=kin_h_err,fmt='D',color=C[1],ms=6,mfc=C[1],label='sim, hub (local inheritance)')
ax.plot(es,nk_h,'D',color=C[1],ms=6,mfc='white',mew=1.5,label='sim, hub (random inheritance)')
ax.plot(es,kin_r,'o',color=C[0],ms=5,label='sim, rim (local)'); ax.plot(es,nk_r,'o',color=C[0],ms=5,mfc='white',mew=1.3,label='sim, rim (random)')
ax.set_xlabel('background mortality $e$'); ax.set_ylabel('evolved $f$')
h,l=ax.get_legend_handles_labels(); o=[0,1,5,2,3,4]; ax.legend([h[i] for i in o],[l[i] for i in o],fontsize=6.5,loc='upper left')
ax.set_title('Law V: pair approximation + best response',fontsize=9,loc='left'); ax.set_xticks(es)
ax.text(-0.14,1.04,'b',transform=ax.transAxes,fontsize=11,fontweight='bold')
fig.tight_layout(); fig.savefig(OUT+'figS2_lawII_graph.pdf',bbox_inches='tight'); fig.savefig(OUT+'figS2_lawII_graph.png',dpi=200,bbox_inches='tight')
