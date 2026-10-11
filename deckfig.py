import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from stories import path
from matplotlib import font_manager as fm
for w in ['Regular','Bold']: fm.fontManager.addfont(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{w}.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':15,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
B='#1f5fae'; O='#b24a1a'; G='#8a8f98'; INK='#1d2a3a'
cap=np.log(2); lo=np.log(0.1)
fig,axs=plt.subplots(1,2,figsize=(16,5.6),gridspec_kw={'width_ratios':[1.35,1]})
fig.patch.set_facecolor('#f7f5ef')
for ax in axs: ax.set_facecolor('#f7f5ef')
ax=axs[0]; rng=np.random.default_rng(65); Z=rng.standard_normal(800); vac=rng.random(800)<0.02
for f_,col,nm in [(0.11,B,'소심 f = 0.11'),(0.6,O,'대담 f = 0.60')]:
    xs,ys,reps,ruin=path(f_,Z,vac); ax.plot(xs,ys,color=col,lw=1.6,label=nm)
    for r in reps: ax.plot(r,cap,'o',ms=9,color=col,zorder=5)
    if ruin: ax.plot(ruin,lo,'X',ms=13,color=col,zorder=5); ax.annotate('기다리다 파산',(ruin,lo),xytext=(14,20),textcoords='offset points',fontsize=15,color=O,fontweight='bold')
for t in np.nonzero(vac)[0]: ax.plot([t+1,t+1],[cap+0.15,cap+0.35],color=G,lw=1.2)
ax.text(5,cap+0.45,'| 빈자리가 난 순간',fontsize=13,color=G)
ax.axhline(cap,color='#1baf7a',ls='--',lw=1.2); ax.axhline(lo,color=G,ls='--',lw=1.2)
ax.text(805,cap,' 천장',va='center',fontsize=13,color='#1baf7a'); ax.text(805,lo,' 파산선',va='center',fontsize=13,color=G)
ax.set_ylim(lo-0.35,cap+0.8); ax.set_xlim(0,800); ax.set_xlabel('세대'); ax.set_ylabel('로그 부'); ax.legend(fontsize=14,loc='lower right',frameon=False)
ax.set_title('기다리는 세상 — 평생 자손: 소심 1.8 · 대담 0.15',fontsize=17,loc='left',color=INK,fontweight='bold')
ax=axs[1]; rng=np.random.default_rng(1027); Z=rng.standard_normal(60); vac=np.ones(60,bool)
for f_,col,nm in [(0.11,B,'소심 f = 0.11'),(0.6,O,'대담 f = 0.60')]:
    xs,ys,reps,ruin=path(f_,Z,vac,60); ax.plot(xs,ys,color=col,lw=1.8,label=nm)
    for r in reps: ax.plot(r,cap,'o',ms=9,color=col,zorder=5)
ax.axvline(60,color=INK,lw=1.4); ax.text(59,lo+0.3,'60세대에\n밀려남',ha='right',fontsize=14,color=INK)
ax.axhline(cap,color='#1baf7a',ls='--',lw=1.2); ax.axhline(lo,color=G,ls='--',lw=1.2)
ax.set_ylim(lo-0.35,cap+0.8); ax.set_xlim(0,64); ax.set_xlabel('세대'); ax.legend(fontsize=14,loc='lower left',frameon=False)
ax.set_title('빼앗는 세상 — 평생 자손: 소심 0.58 · 대담 2.0',fontsize=17,loc='left',color=INK,fontweight='bold')
fig.tight_layout(); fig.savefig('../deck_fig3ab.png',dpi=130,facecolor=fig.get_facecolor())
