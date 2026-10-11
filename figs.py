import numpy as np, pickle, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']; INK='#222'; GREY='#888'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#555','axes.labelcolor':INK,'xtick.color':'#444','ytick.color':'#444','lines.linewidth':1.6,'legend.frameon':False,'font.family':'DejaVu Sans'})
OUT='/mnt/user-data/outputs/figs/'
def save(fig,name): fig.savefig(OUT+name+'.pdf',bbox_inches='tight'); fig.savefig(OUT+name+'.png',dpi=200,bbox_inches='tight'); plt.close(fig)
def lab(ax,s): ax.text(-0.14,1.04,s,transform=ax.transAxes,fontsize=11,fontweight='bold',color=INK)

# ---------- Fig 1: model schematic ----------
from cont import lnparams
rng=np.random.default_rng(4); mu,s=lnparams(0.2,1.0); f=0.25; d=0.01
fig,ax=plt.subplots(figsize=(6.4,2.6)); lw=np.log(1.0); xs=[0]; ys=[lw]; t=0; events=[]
cap=np.log(2); lo=np.log(0.1)
while t<260:
    R=np.exp(mu+s*rng.standard_normal())-1; lw=min(lw+np.log1p(f*R)+np.log(1-d),cap); t+=1
    if lw>=cap and rng.random()<0.08: xs.append(t); ys.append(cap); events.append(t); lw=cap-np.log(2)
    if lw<lo: xs.append(t); ys.append(lo); break
    xs.append(t); ys.append(lw)
ax.plot(xs,ys,color=C[0],lw=1.2)
ax.axhline(cap,color=C[2],ls='--',lw=1); ax.axhline(lo,color=C[1],ls='--',lw=1)
ax.text(262,cap,'  cap $w_{\\rm cap}$: wait for vacancy\n  (reproduce w.p. $v$, split wealth)',va='center',fontsize=8,color=INK)
ax.text(262,lo,'  ruin $w_{\\min}$ (absorbing)',va='center',fontsize=8,color=INK)
for e_ in events: ax.plot([e_],[cap],'o',ms=5,color=C[2])
ax.set_xlim(0,420); ax.set_xticks([0,100,200]); ax.set_xlabel('generation'); ax.set_ylabel('log wealth  ln $w$')
ax.set_title('Each step: $w\\to w(1+fR)(1-\\delta)$; background death w.p. $e$',fontsize=9,loc='left',color=INK)
save(fig,'fig1_model')

# ---------- Fig 2: crowded-limit ESS vs 2delta/m ----------
pts=[(0.100,0.109,0.110,'m=0.2,V=1,δ=0.01'),(0.200,0.202,0.2095,'m=0.2,V=1,δ=0.02'),(0.100,0.095,0.097,'m=0.1,V=1,δ=0.005'),(0.050,0.055,None,'m=0.2,V=1,δ=0.005'),(0.067,0.078,None,'m=0.3,V=1,δ=0.01'),(0.100,0.104,None,'m=0.1,V=0.25,δ=0.005'),(0.040,0.042,None,'m=0.1,V=0.25,δ=0.002'),(0.100,0.110,None,'m=0.2,V=0.25,δ=0.01')]
fig,ax=plt.subplots(figsize=(3.4,3.1))
x=np.array([p[0] for p in pts]); yc=np.array([p[1] for p in pts])
ax.plot([0,0.23],[0,0.23],color=GREY,lw=1,ls='--'); ax.text(0.16,0.145,'$f^*=2\\delta/m$',color=GREY,fontsize=8)
ax.plot(x,yc,'o',ms=6,color=C[0],label='chain ESS')
xs_=[p[0] for p in pts if p[2]]; ys_=[p[2] for p in pts if p[2]]
ax.plot(xs_,ys_,'s',ms=7,mfc='none',mec=C[1],mew=1.5,label='simulation')
ax.set_xlim(0,0.23); ax.set_ylim(0,0.23); ax.set_xlabel('$2\\delta/m$'); ax.set_ylabel('evolved betting fraction $f^*$')
ax.legend(loc='upper left',fontsize=8); ax.set_title('Crowded, ruin-dominated limit\n(Kelly fraction $f^*/f_K$ = 0.08–0.70)',fontsize=9,loc='left')
save(fig,'fig2_crowded')

# ---------- Fig 3: stories + f*(v) and alpha scan ----------
from stories import path
fig,axs=plt.subplots(2,2,figsize=(7.4,5.9))
cap=np.log(2); lo=np.log(0.1)
ax=axs[0,0]; rng=np.random.default_rng(65); Z=rng.standard_normal(800); vac=rng.random(800)<0.02
for f_,col,nm in [(0.11,C[0],'cautious $f$=0.11'),(0.6,C[1],'bold $f$=0.60')]:
    xs,ys,reps,ruin=path(f_,Z,vac); ax.plot(xs,ys,color=col,lw=1.0,label=nm)
    for r in reps: ax.plot(r,cap,'o',ms=6,color=col,zorder=5)
    if ruin: ax.plot(ruin,lo,'X',ms=8,color=col,zorder=5); ax.annotate('ruined while\nwaiting',(ruin,lo),xytext=(10,18),textcoords='offset points',fontsize=7,color=INK)
for t in np.nonzero(vac)[0]: ax.plot([t+1,t+1],[cap+0.12,cap+0.28],color=GREY,lw=0.8)
ax.text(5,cap+0.36,'vacancy opens (v = 0.02)',fontsize=7,color=GREY)
ax.axhline(cap,color=C[2],ls='--',lw=0.8); ax.axhline(lo,color=GREY,ls='--',lw=0.8)
ax.set_ylim(lo-0.3,cap+0.7); ax.set_xlim(0,800); ax.set_xlabel('generation'); ax.set_ylabel('ln $w$'); ax.legend(fontsize=7,loc='lower right')
ax.set_title('Waiting world: survive until a vacancy appears\nmean offspring per life: cautious 1.8, bold 0.15',fontsize=8,loc='left'); lab(ax,'a')
ax=axs[0,1]; rng=np.random.default_rng(1027); Z=rng.standard_normal(60); vac=np.ones(60,bool)
for f_,col,nm in [(0.11,C[0],'cautious $f$=0.11'),(0.6,C[1],'bold $f$=0.60')]:
    xs,ys,reps,ruin=path(f_,Z,vac,60); ax.plot(xs,ys,color=col,lw=1.2,label=nm)
    for r in reps: ax.plot(r,cap,'o',ms=6,color=col,zorder=5)
ax.axvline(60,color=INK,lw=1); ax.text(59,lo+0.25,'displaced\nat $t$=60',ha='right',fontsize=7,color=INK)
ax.axhline(cap,color=C[2],ls='--',lw=0.8); ax.axhline(lo,color=GREY,ls='--',lw=0.8)
ax.set_ylim(lo-0.3,cap+0.7); ax.set_xlim(0,64); ax.set_xlabel('generation'); ax.set_ylabel('ln $w$'); ax.legend(fontsize=7,loc='lower left')
ax.set_title('Displacement world: reach the cap before being pushed out\nmean offspring per life: cautious 0.58, bold 2.0',fontsize=8,loc='left'); lab(ax,'b')
ax=axs[1,0]; res={0.0:(-2.965,0.110),0.005:(-1.154,0.225),0.01:(-0.805,0.290),0.02:(-0.424,0.395)}
for i,e in enumerate([0.0,0.005,0.01,0.02]):
    R=pickle.load(open(f'out/fv_{e}.pkl','rb')); ax.plot(R[:,0],R[:,1],color=C[i],label=f'$e$={e:g}')
    ax.plot(*res[e],'o',ms=8,mfc='white',mec=C[i],mew=1.8)
ax.axhline(0.33,color=GREY,ls='--',lw=1); ax.text(-1.2,0.337,'Kelly',color=GREY,fontsize=8,ha='right')
ax.set_xlabel('$\\log_{10} v$ (reproduction prob. at cap)'); ax.set_ylabel('best response $f^*(v)$'); ax.legend(fontsize=7,loc='upper left',ncol=2); ax.set_ylim(0.09,0.47); lab(ax,'c')
ax.text(-5.9,0.125,'← waiting',fontsize=8,color=GREY); ax.text(-0.1,0.125,'reproduction →',fontsize=8,color=GREY,ha='right')
ax=axs[1,1]; la=np.array([-5,-4,-3.523,-3,-2.523,-2,-1.523,-1,-0.523,-0.222,0]); fa=np.array([0.116,0.140,0.163,0.199,0.250,0.329,0.423,0.530,0.583,0.635,0.635])
ax.plot(la,fa/0.33,'o-',color=C[0],ms=5); ax.axhline(1,color=GREY,ls='--',lw=1); ax.axhline(0.108/0.33,color=GREY,ls=':',lw=1)
ax.text(-5.3,0.108/0.33+0.1,'$\\alpha=0$: $2\\delta/m$',fontsize=8,color=GREY); ax.text(-5.3,1.04,'Kelly',fontsize=8,color=GREY)
ax.set_xlabel('$\\log_{10}\\alpha$ (mortality share of regulation)'); ax.set_ylabel('ESS $f^*/f_K$'); lab(ax,'d')
fig.tight_layout(); save(fig,'fig3_regimes')

# ---------- Fig 4: crossover X and (X,Y) ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
ax=axs[0]; files=['out/univ_0.2_1.0_0.01.pkl','out/univ_0.2_1.0_0.005.pkl','out/univ_0.2_1.0_0.02.pkl','out/univ_0.1_1.0_0.005.pkl','out/univ_0.3_1.0_0.01.pkl','out/univ_0.2_0.25_0.01.pkl']
for i,fn in enumerate(files):
    D=pickle.load(open(fn,'rb')); X=np.array([o[0] for o in D['out']]); F=np.array([o[2] for o in D['out']])/D['f0']
    ax.plot(X,F,'o-',ms=3.5,color=C[i],label=f"m={D['m']},V={D['V']:g},δ={D['d']} ($T_r$={D['Tr0']:.0e})")
ax.set_xscale('log'); ax.axvline(1,color=GREY,ls='--',lw=1); ax.set_xlabel('$X=e\\,T_r$'); ax.set_ylabel('$f^*(e)/f^*(0)$'); ax.legend(fontsize=6,loc='upper left'); lab(ax,'a')
ax=axs[1]; P=pickle.load(open('out/xy.pkl','rb'))
Xs=np.array([p[1] for p in P]); Ys=np.array([p[2] for p in P]); Rr=np.array([p[3] for p in P]); m_=Xs>=1
sc=ax.scatter(Xs[m_],Ys[m_],c=Rr[m_],cmap='Blues',vmin=0.8,vmax=3.1,s=28,edgecolors='#333',linewidths=0.4)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('$X=e\\,T_r$'); ax.set_ylabel('$Y=e\\,t_c$'); cb=fig.colorbar(sc,ax=ax); cb.set_label('$f^*(e)/f^*(0)$'); lab(ax,'b')
fig.tight_layout(); save(fig,'fig4_crossover')

# ---------- Fig 5: freezing ----------
Cg=pickle.load(open('out/cgrid_0.005.pkl','rb'))
beta=lambda x: np.interp(x,Cg[:,0],Cg[:,2]); uu=lambda x: np.exp(np.interp(x,Cg[:,0],np.log(Cg[:,1]))); cc=lambda x: np.interp(x,Cg[:,0],Cg[:,4])
def theo(sig,T=1e8):
    f=0.2;G=0.0;t=0.0;ts=[];fs=[]
    while t<T:
        dt=10.0 if t<1e6 else t*2e-4
        f+=dt*G*beta(f); G=max(G+dt*(sig**2*uu(f)+G**2*cc(f)),0); t+=dt; ts.append(t); fs.append(f)
    return np.array(ts),np.array(fs)
fig,ax=plt.subplots(figsize=(4.4,3.0))
for i,(sig,fn) in enumerate([(0.005,'out/frz_0.2_0.005_0.2_0.005_11.pkl'),(0.0025,'out/frz_0.2_0.005_0.2_0.0025_13.pkl')]):
    h=pickle.load(open(fn,'rb')); ax.plot(h[1:,0],h[1:,2],color=C[i],lw=1,alpha=0.8,label=f'simulation, σ={sig}')
    ts,fs=theo(sig); ax.plot(ts,fs,'--',color=C[i],lw=1.6,label=f'theory, σ={sig}')
ax.axhline(0.055,color=GREY,ls=':',lw=1); ax.text(1.2e3,0.06,'ESS 0.055',fontsize=8,color=GREY)
ax.set_xscale('log'); ax.set_xlim(1e3,1e8); ax.set_xlabel('generation $t$'); ax.set_ylabel('mean betting fraction $\\bar f$'); ax.legend(fontsize=7)
ax.set_title('Evolutionary freezing (m=0.2, δ=0.005, N=4096)',fontsize=9,loc='left')
save(fig,'fig5_freezing')

# ---------- Fig 6: extinction and graph ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
ax=axs[0]
for i,(fn,lab_,erg,ramp) in enumerate([('out/ec_0.2_0.01.npy','m=0.2, δ=0.01',0.622,[(0.66,0.0401),(0.63,0.0403)]),('out/ec_0.1_0.005.npy','m=0.1, δ=0.005',0.198,[(0.317,0.0101)])]):
    E=np.load(fn); ax.plot(E[:,0],E[:,1],color=C[i],label=lab_); ax.axvline(erg,color=C[i],ls=':',lw=1)
    for fr,er in ramp: ax.plot(fr,er,'s',ms=6,mfc='none',mec=C[i],mew=1.5)
ax.set_xlabel('fixed betting fraction $f$'); ax.set_ylabel('extinction threshold $e_c(f)$'); ax.legend(fontsize=7.5)
lab(ax,'a')
ax=axs[1]
for i,e in enumerate([0.005,0.01]):
    R=pickle.load(open(f'out/fv_{e}.pkl','rb')); ax.plot(R[:,0],R[:,1],color=C[i+1],label=f'$f^*(v)$, $e$={e}')
pts6={0.005:[(0.07,0.23,'rim'),(0.65,0.275,'hub')],0.01:[(0.16,0.295,'rim'),(0.93,0.385,'hub')]}
for i,e in enumerate([0.005,0.01]):
    for v,fs,nm in pts6[e]:
        ax.plot(np.log10(v),fs,'o' if nm=='rim' else 'D',ms=7,mfc='white',mec=C[i+1],mew=1.6)
        ax.annotate(nm,(np.log10(v),fs),xytext=(4,4),textcoords='offset points',fontsize=7.5,color=INK)
ax.set_xlim(-2.5,0.1); ax.set_ylim(0.15,0.45); ax.set_xlabel('$\\log_{10}$ measured local $v$'); ax.set_ylabel('evolved $f$ at position'); ax.legend(fontsize=7.5,loc='upper left'); lab(ax,'b')
fig.tight_layout(); save(fig,'fig6_extinction_graph')
print('ok')
