import pickle,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']; MK='osD^vP'; INK='#222'; GR='#888'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'font.family':'DejaVu Sans'})
R=pickle.load(open('family.pkl','rb'))
lab={'base':'$Q=2,\\ b=2.3$ (reference)','Q4':'$Q=4,\\ b=2.3$','Q1':'$Q=1,\\ b=2.3$','b3.0':'$Q=2,\\ b=3.0$','b1.6':'$Q=2,\\ b=1.6$','mV':'$Q=2,\\ b=2.3$; $m=0.1$, Var$R=0.25$'}
chk=[]
try:
    for L in open('famcheck.log'):
        p=L.split(); chk.append((p[0],float(p[1]),float(p[4])))
except Exception: pass
fig,axs=plt.subplots(1,3,figsize=(10.4,3.1),gridspec_kw={'width_ratios':[1,1,0.9]})
for i,r in enumerate(R):
    a=np.array([(mu,f) for mu,f,v in r['rows'] if np.isfinite(f)]); y=(a[:,1]-r['f0'])/(r['fmax']-r['f0'])
    X=np.maximum(a[:,0]*r['Tr0'],1e-2)
    axs[0].plot(X,y,'-',color=C[i],lw=1.5,marker=MK[i],ms=3.5,markevery=3,label=lab[r['name']])
    axs[0].plot([r['muc']*r['Tr0']],[1],MK[i],color=C[i],ms=6)
    axs[1].plot(a[:,0]/r['muc'],y,'-',color=C[i],lw=1.5,marker=MK[i],ms=3.5,markevery=3)
    for n,u,f in chk:
        if n==r['name']: axs[1].plot([u],[(f-r['f0'])/(r['fmax']-r['f0'])],MK[i],color=C[i],ms=8,mfc='white',mew=1.5,zorder=5)
ax=axs[0]; ax.set_xscale('log'); ax.axvline(1,color=GR,ls='-.',lw=0.8); ax.text(1.15,0.45,'$X=1$',color=INK,fontsize=8)
ax.set_xlabel('$X=\\mu_{\\rm ext}T_r$'); ax.set_ylabel('$(f^*-f_{\\min})/(f_{\\max}-f_{\\min})$'); ax.set_xlim(1e-2,2e6); ax.set_ylim(-0.03,1.05)
ax.text(-0.16,1.03,'a',transform=ax.transAxes,fontsize=11,fontweight='bold')
ax.set_title('onset at $X\\approx1$; edge at $X_{\\max}\\propto e^{D(Q-1)}$',fontsize=8.5,loc='left')
ax=axs[1]; ax.set_xlabel('$\\varepsilon=\\mu_{\\rm ext}/\\mu_c$'); ax.set_xlim(0,1.02); ax.set_ylim(-0.03,1.05)
ax.plot([0,1],[0,1],':',color=GR,lw=0.8)
h,l=axs[0].get_legend_handles_labels(); ax.legend(h,l,fontsize=6.6,loc='lower right')
ax.plot([],[],'o',color=GR,mfc='white',mew=1.5,ms=7,label='_')
ax.text(0.03,0.93,'open: population ESS (chain)',fontsize=7,color=INK,transform=ax.transAxes)
ax.text(-0.16,1.03,'b',transform=ax.transAxes,fontsize=11,fontweight='bold'); ax.set_title('near the edge the curve depends on $\\varepsilon$ only',fontsize=8.5,loc='left')
ph=pickle.load(open('phase.pkl','rb')); Q=np.array([r[0] for r in ph]); Xon=np.array([r[1] for r in ph]); XK=np.array([r[2] for r in ph]); Xm=np.array([r[3] for r in ph])
ax=axs[2]; lo=1e-2; hi=3e6
ax.fill_between(Q,lo,XK,color='#cde2fb',lw=0); ax.fill_between(Q,XK,Xm,color='#fbd9c9',lw=0); ax.fill_between(Q,Xm,hi,color='#e6e6e3',lw=0)
ax.plot(Q,Xon,'o-',color=C[0],ms=4,lw=1.3,label='onset ($f^*/f_{\\min}=1.05$)'); ax.plot(Q,XK,'s-',color=INK,ms=4,lw=1.3,label='Kelly crossing'); ax.plot(Q,Xm,'D-',color=C[1],ms=4,lw=1.3,label='extinction edge')
qq=np.linspace(0.7,4,10); ax.plot(qq,np.exp(3.0*(qq-1)+np.log(Xm[1])),':',color=GR,lw=1)
ax.set_yscale('log'); ax.set_ylim(lo,hi); ax.set_xlim(0.7,4); ax.set_xlabel('$Q=m^2/(2\\delta\\,\\mathrm{Var}R)$'); ax.set_ylabel('$X$')
ax.text(3.0,0.05,'below Kelly',fontsize=8,color='#184f95',ha='center'); ax.annotate('above Kelly',(3.5,1.5e5),xytext=(3.3,4e2),fontsize=8,color='#8a3a14',ha='center',arrowprops=dict(arrowstyle='->',color='#8a3a14',lw=0.8)); ax.text(3.2,1.2e6,'extinct',fontsize=8,color='#555',ha='center'); ax.legend(fontsize=6.5,loc='upper left')

ax.set_title('phase diagram ($b=2.3$)',fontsize=8.5,loc='left'); ax.text(-0.2,1.03,'c',transform=ax.transAxes,fontsize=11,fontweight='bold')
fig.tight_layout()
for e in ['pdf','png']: fig.savefig(f'/mnt/user-data/outputs/figs/fig_families.{e}',dpi=200,bbox_inches='tight')
