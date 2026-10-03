import sys, numpy as np, warnings; warnings.simplefilter('ignore')
from common import *; from level import *; from rowodo import *; from vo import load_vo
from exp_vo import rows_and_sigs, match, second_best, RESM
t,pos,rot,imgs,rimgs=load_seq('2023-12-26-13-39-43')
obs=FrameObs(imgs,level_rotation(np.array([0.0129,-0.9569,-0.2901])),1.25)
vo,odo=load_vo('1226_1339')
sp=float(sys.argv[1]) if len(sys.argv)>1 else 0.505
def win(k_end,back,fwd,s0,s1): return [k for k in range(s0,s1+1,2) if odo[k_end]-back<=odo[k]<=odo[k_end]+fwd]
th=np.radians(1.09); dd=np.array([np.cos(th),np.sin(th)])
for q in (19950,20050,20150):
    A=np.arange(12182,13669); U=pos[:, :2]@dd
    c=int(A[np.argmin(abs(U[A]-U[q]))])
    fq=win(q,10,0,19858,20221); fm=win(c,20,10,12182,13669)
    Uq,Wq,Pq,_=row_track(obs,fq,odo,sp); Um,Wm,Pm,_=row_track(obs,fm,odo,sp)
    bq=local_bev(obs,fq,Uq,Wq,Pq,(-1,Uq[-1]+4),(-4,4)); bm=local_bev(obs,fm,Um,Wm,Pm,(-1,Um[-1]+4),(-4,4))
    cols=np.where(np.isfinite(bq).mean(0)>0.3)[0]; bq=bq[:,cols[0]:cols[-1]+1]; uq0=-1+cols[0]*RESM
    wsq,sq=rows_and_sigs(bq,uq0,-4,np.median(Wq)); wsm,sm=rows_and_sigs(bm,-1,-4,np.median(Wm))
    H=match(dict(ws=wsq,sigs=sq,u0=uq0),dict(ws=wsm,sigs=sm,u0=-1),sp)
    jc=int(np.argmin(abs(np.array(fm)-c)))
    dmap=dd; lmap=np.array([-dd[1],dd[0]]); gu=(pos[q,:2]-pos[c,:2])@dmap; gw=(pos[q,:2]-pos[c,:2])@lmap
    print('q',q,'rows q',np.round(wsq,2),'rows m',np.round(wsm,2),' GT q-c: u %.2f w %.2f'%(gu,gw))
    for h in H[:5]:
        s,k,tu,tw=h[1:5]; pu=s*Uq[-1]+tu-Um[jc]; pw=s*Wq[-1]+tw-Wm[jc]
        print('   z=%.2f flip=%d k=%d -> q en mapa u=%.2f w=%.2f (n=%d)'%(h[0],s,k,pu,pw,h[5]))
