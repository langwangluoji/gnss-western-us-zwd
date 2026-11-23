"""Reproduce metadata-only station proposal and vector research diagrams."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p/'AGENTS.md').exists())
SOURCE = ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/02_data/processed/gnss_field_server_v1/stations.csv'
SEED = 20261123


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_split():
    df = pd.read_csv(SOURCE)
    ab = df[df.role.isin(['input', 'train'])].copy()
    # Close physical groups and 100 km blocks before assigning any fold.
    ids = ab.index.tolist(); parent = {i:i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b: parent[max(a,b)]=min(a,b)
    ab['block_x']=np.floor(ab.x/100000).astype(int)
    ab['block_y']=np.floor(ab.y/100000).astype(int)
    for cols in [['physical_group'],['block_x','block_y']]:
        for _,g in ab.groupby(cols):
            a=g.index[0]
            for b in g.index[1:]: union(a,b)
    ab['unit']=[find(i) for i in ids]
    ab['height_bin']=np.digitize(ab.height,[500,1500])
    units=[]
    for uid,g in ab.groupby('unit'):
        h=np.bincount(g.height_bin,minlength=3).astype(float)
        key=hashlib.sha256(f'{SEED}:{uid}'.encode()).hexdigest()
        units.append((uid,len(g),h,key))
    units.sort(key=lambda v:(-v[1],v[3]))
    totals=np.zeros(5); heights=np.zeros((5,3))
    target=len(ab)/5; ht=np.bincount(ab.height_bin,minlength=3)/5
    assigned={}
    for uid,n,h,key in units:
        costs=[]
        for f in range(5):
            t=totals.copy();z=heights.copy();t[f]+=n;z[f]+=h
            cost=np.sum(((t-target)/target)**2)+.25*np.sum(((z-ht)/np.maximum(ht,1))**2)
            costs.append(cost)
        f=int(np.argmin(costs));assigned[uid]=f;totals[f]+=n;heights[f]+=h
    ab['fold']=[assigned[u] for u in ab.unit]
    ab['new_role']=np.where(ab.fold==0,'confirmation',np.where(ab.fold==1,'validation','training'))
    result=df.copy();result['new_role']=result.role.map({'development':'legacy_C','retrospective':'legacy_D'})
    for c in ['block_x','block_y','unit','fold']:
        result[c]=pd.Series(index=result.index,dtype='Int64');result.loc[ab.index,c]=ab[c].astype('Int64')
    result.loc[ab.index,'new_role']=ab.new_role
    assert len(ab)==685 and result.new_role.notna().all()
    assert ab.groupby('physical_group').fold.nunique().max()==1
    assert ab.groupby(['block_x','block_y']).fold.nunique().max()==1
    # Cross-role co-locations require explicit exclusion before data build.
    cross=set(ab.physical_group)&set(df.loc[~df.index.isin(ab.index),'physical_group'])
    assert not cross, f'Physical groups shared with C/D: {cross}'
    result.to_csv(OUT/'station_assignment_proposal.csv',index=False,lineterminator='\n')
    summary={'status':'METADATA_ONLY_PROPOSAL_NOT_ACTIVATED','seed':SEED,'block_m':100000,
             'source_sha256':digest(SOURCE),'script_sha256':digest(Path(__file__)),
             'counts':result.new_role.value_counts().to_dict(),'fold_counts':ab.fold.value_counts().sort_index().to_dict(),
             'old_roles_by_new':pd.crosstab(result.new_role,result.role).to_dict(),
             'physical_overlap':False,'labels_read':False,'weather_values_read':False,
             'height_counts':pd.crosstab(ab.new_role,ab.height_bin).to_dict(),
             'note':'QC may reduce effective counts; repartition does not erase prior development exposure.'}
    (OUT/'station_split_receipt.json').write_bytes(json.dumps(summary,ensure_ascii=False,indent=2).encode('utf-8'))
    return result,summary


def save(fig,name):
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    svg=OUT/f'{name}.svg'
    svg.write_bytes(b'\n'.join(line.rstrip() for line in svg.read_bytes().splitlines())+b'\n')
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight')
    plt.close(fig)


def diagrams(df,summary):
    plt.rcParams.update({'font.family':'Microsoft YaHei','svg.fonttype':'none','font.size':11})
    fig,ax=plt.subplots(figsize=(14,10));ax.set(xlim=(0,14),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,title,body,color='#e8f0fa'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc=color,ec='#456078',lw=1.2))
        ax.text(x+w/2,y+h-.28,title,ha='center',va='top',fontsize=12,fontweight='bold',color='#152e42')
        ax.text(x+w/2,y+h-.68,body,ha='center',va='top',fontsize=10,linespacing=1.6,color='#223744')
    def arrow(a,b,label=''):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#456078',lw=1.5))
        if label:ax.text((a[0]+b[0])/2+.1,(a[1]+b[1])/2,label,fontsize=9,color='#456078')
    ax.text(.15,9.75,'地形感知条件图扩散：动态站网 → 连续ZWD序列',fontsize=19,fontweight='bold',color='#152e42')
    box(.3,7.55,4,1.55,'① 固定外层划分','旧A+B共685站 → 5个空间折\n训练 / 验证 / 确认；C/D保留登记角色')
    box(4.95,7.55,4,1.55,'② 训练池内动态遮站','选时间窗口与空间邻域\n条件站与目标站互不重叠')
    box(9.6,7.55,4,1.55,'③ 输入与目标','条件：ZWD、T/P、地形、掩码\n监督：目标站真实ZWD；直接生成')
    arrow((4.38,8.3),(4.85,8.3));arrow((9.03,8.3),(9.5,8.3))
    box(.3,4.9,4,1.7,'④ 条件编码器','地形图注意力 ×2\n因果时间Transformer ×2\n输出观测记忆 Cτ', '#e6f3ef')
    box(4.95,4.9,4,1.7,'⑤ 查询节点去噪器','带噪ZWD + 查询地形 + 扩散步\n空间图交互 + 观测交叉注意力\n4个模块；输出 vθ', '#e6f3ef')
    box(9.6,4.9,4,1.7,'⑥ 扩散训练 / 采样','训练：真实ZWD加噪 → v目标\n生成：高斯噪声 → 20步DDIM\n输出当前小时联合ZWD成员', '#e6f3ef')
    arrow((11.6,7.48),(11.6,6.7));arrow((6.95,7.48),(6.95,6.7))
    arrow((4.38,5.75),(4.85,5.75));arrow((9.03,5.75),(9.5,5.75))
    ax.plot([11.6,11.6,2.3],[7.48,7.05,7.05],color='#456078',lw=1.3)
    arrow((2.3,7.05),(2.3,6.7),'条件流')
    box(.3,2.1,4,1.75,'⑦ 同成员历史状态','当前生成 → 历史编码 → 下一小时\n训练：冷启动 / teacher forcing\n微调及评价：自身生成历史', '#fff1dc')
    box(4.95,2.1,4,1.75,'⑧ 真实观测监督','基础：有效目标上的 v-MSE\n微调：CRPS + 增量CRPS\nK4 × 4小时可微生成', '#fff1dc')
    box(9.6,2.1,4,1.75,'⑨ 输出与检验','固定查询集的24/72小时成员序列\nRMSE / CRPS / 时空关联\n查询扩展通过后进入格网研究', '#fff1dc')
    arrow((11.6,4.82),(11.6,3.95));arrow((6.95,4.82),(6.95,3.95))
    ax.plot([11.6,11.6,2.3],[2.02,1.6,1.6],color='#456078',lw=1.3)
    arrow((2.3,1.6),(2.3,2.02));arrow((4.38,3.65),(5.15,4.8))
    ax.text(.4,.95,'两条时间轴：物理小时 τ 顺序推进；每个小时内部执行扩散去噪步 s。',fontsize=12,color='#152e42')
    ax.text(.4,.5,'方案图 · 尚未训练本架构 · 无外接均值场 · 隐藏站真实记录用于监督 · 固定外层隔离始终保留',fontsize=10,color='#456078')
    save(fig,'architecture')
    fig,ax=plt.subplots(figsize=(9,9))
    styles=[('training','训练池','#237b60','o'),('validation','验证','#d59421','s'),('confirmation','空间确认','#3567aa','^'),('legacy_C','原C','#999999','x'),('legacy_D','原D','#c3c3c3','+')]
    for role,label,c,m in styles:
        s=df[df.new_role==role]
        ax.scatter(s.x/1000,s.y/1000,s=17,c=c,marker=m,alpha=.8,label=f'{label} ({len(s)})')
    ax.set(xlabel='投影东向坐标 / km',ylabel='投影北向坐标 / km',title='站点空间划分提案：100 km块 + 物理组整体分配')
    ax.set_aspect('equal');ax.legend(loc='upper left',fontsize=9);ax.grid(alpha=.18)
    fig.text(.12,.02,'仅使用站点元数据；未读取观测标签。新划分不改变历史开发暴露事实。',fontsize=9)
    save(fig,'station_split')
    fig,ax=plt.subplots(figsize=(15,7));ax.set(xlim=(0,15),ylim=(0,7));ax.axis('off')
    ax.text(.2,6.7,'去噪器内部：当前空间场 × 生成历史 × 可见观测',fontsize=18,fontweight='bold',color='#152e42')
    box(.25,4.65,4.1,1.2,'查询输入','带噪ZWD + 地形位置 + 时间\n聚合气象 / 支持度 → 64维')
    box(5.4,4.65,4.1,1.2,'生成历史编码','同成员前6h ZWD / 有效掩码\nGRU(hidden=64) → 历史向量')
    box(10.55,4.65,4.1,1.2,'条件记忆','可见站12h观测\n图编码 + 因果Transformer')
    box(.25,1.85,2.8,1.6,'扩散步调制','s的正弦嵌入\nMLP → AdaLN\n当前状态 hq', '#e6f3ef')
    box(3.7,1.85,2.8,1.6,'交叉注意力','查询读取可见站\n位移 / 高差边偏置\n残差连接', '#e6f3ef')
    box(7.15,1.85,2.8,1.6,'查询图交互','当前目标节点互传\n12邻居 + 自环\n空间联合生成', '#e6f3ef')
    box(10.6,1.85,2.8,1.6,'前馈更新','LayerNorm\n64 → 128 → 64\nSiLU + 残差', '#e6f3ef')
    arrow((3.13,2.65),(3.6,2.65));arrow((6.58,2.65),(7.05,2.65));arrow((10.03,2.65),(10.5,2.65))
    arrow((2.3,4.55),(1.65,3.55));arrow((7.45,4.55),(2.8,3.55))
    ax.plot([12.6,12.6,5.1],[4.55,4.05,4.05],color='#456078',lw=1.3)
    arrow((5.1,4.05),(5.1,3.55))
    ax.text(6.8,1.28,'上面四步组成一个模块，重复4次 → LayerNorm + Linear(64,1) → v预测',ha='center',fontsize=12,color='#152e42')
    ax.text(.3,.55,'v预测 → 还原干净ZWD与噪声 → DDIM前进一步；20次后输出当前小时ZWD，进入同成员历史缓冲。',fontsize=10,color='#456078')
    save(fig,'denoiser')


if __name__=='__main__':
    df,summary=make_split();diagrams(df,summary)
    print(json.dumps(summary,ensure_ascii=False))
