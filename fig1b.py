import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Ellipse
from cont import lnparams
C=['#2a78d6','#eb6834','#1baf7a']; INK='#222'; GREY='#777'; BLUE='#1f5fae'; ORG='#b24a1a'
plt.rcParams.update({'font.size':8.5,'font.family':'DejaVu Sans'})
fig=plt.figure(figsize=(7.4,3.3))
axA=fig.add_axes([0.0,0.0,0.52,0.93]); axA.set_xlim(0,10); axA.set_ylim(0,7); axA.axis('off')
axB=fig.add_axes([0.60,0.14,0.39,0.74])
axA.text(0.1,6.75,'a',fontsize=11,fontweight='bold'); axB.text(-0.2,1.08,'b',transform=axB.transAxes,fontsize=11,fontweight='bold')
# wealth column
axA.add_patch(FancyBboxPatch((0.5,0.9),1.6,4.9,boxstyle='round,pad=0.05,rounding_size=0.25',fc='#eef2f7',ec='#8a96a6',lw=1))
axA.add_patch(Rectangle((0.5,5.35),1.6,0.45,fc=C[2],ec='none')); axA.text(1.3,5.575,'cap',ha='center',va='center',color='white',fontsize=8,fontweight='bold')
axA.add_patch(Rectangle((0.5,0.9),1.6,0.45,fc=C[1],ec='none')); axA.text(1.3,1.125,'ruin',ha='center',va='center',color='white',fontsize=8,fontweight='bold')
axA.add_patch(Rectangle((0.75,1.35),1.1,2.3,fc='#9fbfe6',ec='none')); axA.text(1.3,2.5,'wealth\n$w$',ha='center',va='center',fontsize=8,color=INK)
axA.annotate('',xy=(2.35,4.3),xytext=(2.35,3.0),arrowprops=dict(arrowstyle='<->',color=INK,lw=1))
axA.text(2.5,3.65,'stake $f w$\n× return $R$\n− cost $\\delta$',fontsize=7.5,va='center',color=INK)
axA.text(1.3,0.35,'individual',ha='center',fontsize=8,color=GREY)
axA.text(0.5,6.25,'each generation: $w\\to w(1+fR)(1-\\delta)$',fontsize=8,color=INK)
# arrow from cap to two worlds
axA.annotate('',xy=(4.5,5.1),xytext=(2.2,5.575),arrowprops=dict(arrowstyle='->',color=C[2],lw=1.3))
axA.text(2.6,5.0,'reproduce',fontsize=7.5,color=INK,rotation=-11)
def grid(x0,y0,occ,hl=None,mark=None):
    s=0.62
    for i in range(3):
        for j in range(3):
            k=i*3+j; fc='#c9d6e6' if occ[k] else 'white'
            if hl==k: fc='#ffe0cc'
            axA.add_patch(Rectangle((x0+j*s,y0+(2-i)*s),s*0.92,s*0.92,fc=fc,ec='#8a96a6',lw=0.8))
            if occ[k]: axA.add_patch(Ellipse((x0+j*s+s*0.46,y0+(2-i)*s+s*0.46),0.32,0.32*(0.52*7.4/10)/(0.93*3.3/7),fc=BLUE if k!=mark else ORG,ec='none'))
    return s
# birth regulation world
axA.text(4.3,5.8,'birth regulation (vacancy)',fontsize=8,fontweight='bold',color=BLUE)
occ=[1,1,1,1,1,1,1,1,0]; s=grid(4.5,3.55,occ,hl=8,mark=4)
axA.annotate('',xy=(4.5+2*s+0.28,3.55+0.28),xytext=(4.5+s+0.28,3.55+s+0.28),arrowprops=dict(arrowstyle='->',color=ORG,lw=1.2))
axA.text(6.55,4.75,'offspring waits\nfor an empty site\n→ survive while\n    waiting',fontsize=7.5,va='center',color=INK)
# mortality regulation world
axA.text(4.3,2.95,'mortality regulation (lottery)',fontsize=8,fontweight='bold',color=ORG)
occ2=[1]*9; s=grid(4.5,0.7,occ2,hl=2,mark=4)
axA.annotate('',xy=(4.5+2*s+0.28,0.7+2*s+0.28),xytext=(4.5+s+0.28,0.7+s+0.28),arrowprops=dict(arrowstyle='->',color=ORG,lw=1.2))
axA.text(4.5+2*s+0.28,0.7+2*s+0.28,'×',fontsize=12,ha='center',va='center',color=INK)
axA.text(6.55,1.6,'offspring displaces\nan occupant\n→ be ready fast',fontsize=7.5,va='center',color=INK)
axA.text(4.3,0.2,'background death with prob. $e$ in both',fontsize=7.5,color=GREY)
# panel b: trajectory
rng=np.random.default_rng(4); mu,sd=lnparams(0.2,1.0); f=0.25; d=0.01
lw=0.0; xs=[0]; ys=[lw]; t=0; ev=[]; cap=np.log(2); lo=np.log(0.1)
while t<260:
    R=np.exp(mu+sd*rng.standard_normal())-1; lw=min(lw+np.log1p(f*R)+np.log(1-d),cap); t+=1
    if lw>=cap and rng.random()<0.08: xs.append(t); ys.append(cap); ev.append(t); lw=cap-np.log(2)
    if lw<lo: xs.append(t); ys.append(lo); break
    xs.append(t); ys.append(lw)
axB.plot(xs,ys,color=C[0],lw=1.1)
axB.axhline(cap,color=C[2],ls='--',lw=1); axB.axhline(lo,color=C[1],ls='--',lw=1)
for e_ in ev: axB.plot(e_,cap,'o',ms=4.5,color=C[2])
axB.text(3,cap+0.12,'cap: wait, reproduce w.p. $v$',fontsize=7.5,color=INK); axB.text(3,lo+0.12,'ruin (absorbing)',fontsize=7.5,color=INK)
axB.set_ylim(lo-0.25,cap+0.55); axB.set_xlabel('generation'); axB.set_ylabel('log wealth  ln $w$')
for sp in ['top','right']: axB.spines[sp].set_visible(False)
axB.text(0.99,0.5,'dots: reproduction\n(wealth halves)',transform=axB.transAxes,ha='right',fontsize=7,color=GREY)
fig.savefig('/mnt/user-data/outputs/figs/fig1_model.pdf',bbox_inches='tight'); fig.savefig('/mnt/user-data/outputs/figs/fig1_model.png',dpi=200,bbox_inches='tight'); print('ok')
