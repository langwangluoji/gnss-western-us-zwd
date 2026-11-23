"""Post-training matched marginal-preserving temporal ablation and report."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sequence_mechanism_spike import ROOT as V1, fair_crps, MeanModel, RandomField, encode_inputs
from sequence_grid_spike import GridMean,GridRandom,grid_inputs,XY,TERRAIN,QI,QW,STORE
from gnss_field.artifacts import write_json,sha256

ROOT=V1.parent/'20261123_sequence_spike_v02_grid'


def crps_report(s,y,m):
    return dict(crps=float(fair_crps(s,y,m)),increment_crps=float(fair_crps(s[:,:,1:]-s[:,:,:-1],y[:,1:]-y[:,:-1],m[:,1:]&m[:,:-1])),
        station_order_pair_crps=float(fair_crps(s[:,:,:,1:]-s[:,:,:,:-1],y[:,:,1:]-y[:,:,:-1],m[:,:,1:]&m[:,:,:-1])))


def main():
    torch.set_num_threads(2)
    r=json.loads((ROOT/'results.json').read_text()); a=dict(np.load(ROOT/'data.npz'))
    receipt=json.loads((ROOT/'data_receipt.json').read_text())
    assert sha256(ROOT/'data.npz')==receipt['data_sha256']
    assert sha256(ROOT/'PROTOCOL.md')==receipt['protocol_sha256']
    from sequence_mechanism_spike import __file__ as core_file
    from sequence_grid_spike import __file__ as adapter_file
    assert sha256(core_file)==r['runtime']['source_sha256']==receipt['source_code']
    operator=json.loads((ROOT/'operator_receipt.json').read_text())
    assert sha256(adapter_file)==operator['adapter_sha256']
    old_receipt=json.loads((V1/'data_receipt.json').read_text())
    assert sha256(V1/'source_snapshot.py')==old_receipt['source_code']
    assert sha256(V1/'PROTOCOL.md')==old_receipt['protocol_sha256']
    for name,digest in r['runtime']['checkpoints'].items():assert sha256(ROOT/name)==digest
    diagnostics={}; all_pred={}
    for seed in ('42','17'):
        diagnostics[seed]={}
        for arm in ('independent','ar'):
            p=dict(np.load(ROOT/f'{arm}_{seed}_predictions.npz'))
            s=torch.tensor(p['samples']); y=torch.tensor(p['target']); m=torch.tensor(p['mask'])
            original=crps_report(s,y,m)
            assert abs(original['crps']-r[seed][arm]['outer']['crps'])<1e-5
            shuffled=s.clone(); rng=np.random.default_rng(883)
            for b in range(s.shape[0]):
                for t in range(s.shape[2]):shuffled[b,:,t]=s[b,rng.permutation(s.shape[1]),t]
            shuffled_score=crps_report(shuffled,y,m)
            assert torch.equal(s.sort(dim=1).values,shuffled.sort(dim=1).values)
            perseq=[]
            for b in range(s.shape[0]):
                perseq.append(crps_report(s[b:b+1],y[b:b+1],m[b:b+1]))
            diagnostics[seed][arm]=dict(original=original,temporal_member_shuffle=shuffled_score,
                marginal_samples_exactly_preserved=True,per_sequence=perseq)
            all_pred[(seed,arm)]=p
        for metric in ('crps','increment_crps'):
            delta=np.array([v[metric] for v in diagnostics[seed]['ar']['per_sequence']])-np.array([v[metric] for v in diagnostics[seed]['independent']['per_sequence']])
            rng=np.random.default_rng(20261123); boot=delta[rng.integers(0,12,(5000,12))].mean(1)
            diagnostics[seed][metric+'_paired_blocks']=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    # Adapter equality against independent evaluation at the four explicit grid nodes.
    subset=a['split_outer'][:8]
    fx,fb,_=grid_inputs(a,a['outer_raw'][:1],a['outer_hours'][:1],a['xy'],a['terrain'])
    x=torch.tensor(fx[:,:,subset]); gm=GridMean(fx.shape[-1],True)
    gm.load_state_dict(torch.load(ROOT/'gru_17.pt',map_location='cpu',weights_only=True)); gm.eval()
    plain=MeanModel(fx.shape[-1]-1,True); plain.load_state_dict(gm.state_dict());plain.eval()
    with torch.no_grad():
        residual,h=gm(x)
        raw_r,raw_h=plain(x[...,:-1].flatten(2,3))
        explicit=(raw_r.reshape(1,30,8,4)*x[...,-1]).sum(-1)
        assert torch.allclose(residual,explicit,atol=1e-6)
        changed=x.clone();changed[:,15:,:,:,:-1]+=10
        assert torch.equal(gm(x)[1][:,:15],gm(changed)[1][:,:15])
        gg=GridRandom(True);gg.load_state_dict(torch.load(ROOT/'ar_17.pt',map_location='cpu',weights_only=True))
        plain_g=RandomField(True);plain_g.load_state_dict(gg.state_dict())
        torch.manual_seed(77);noise=torch.randn(1,8,30,8)
        expected=(plain_g(raw_h,noise).reshape(1,8,30,8,4)*x[...,-1][:,None]).sum(-1)
        assert torch.allclose(gg(h,noise),expected,atol=1e-5)
        chunked=torch.cat([gg(h[:,:,:3],noise),gg(h[:,:,3:],noise)],-1)
        assert torch.allclose(chunked,gg(h,noise),atol=1e-5)
        # Generate canonical full-grid sequence with exactly the trained parameters.
        torch.manual_seed(9301);noise=torch.randn(1,8,30,8);chunks=[]
        for start in range(0,len(XY),512):
            stop=start+512
            gx,gb,_=encode_inputs(a,a['outer_raw'][:1],a['outer_hours'][:1],XY[start:stop],TERRAIN[start:stop])
            gr,gh=plain(torch.tensor(gx))
            fields=torch.tensor(gb)[:,None]+gr[:,None]+plain_g(gh,noise)
            chunks.append(fields[0,:,6:].numpy())
        field=np.concatenate(chunks,-1).reshape(8,24,144,96)
        assert np.isfinite(field).all()
        # Independent full-grid round trip checks node indexing, not just head aggregation.
        roundtrip=(field.reshape(8,24,-1)[:,:,QI[subset]]*QW[subset]).sum(-1)
        factor=np.exp(-a['heights'][subset]/3000)
        roundtrip*=factor
        adapter=(torch.tensor(fb[:,:,subset])[:,None]+residual[:,None]+gg(h,noise))[0,:,6:].numpy()*factor
        roundtrip_error=float(np.max(np.abs(roundtrip-adapter)))
        assert roundtrip_error<2e-4, roundtrip_error
        np.savez_compressed(ROOT/'example_grid_sequence.npz',Z0_members=field,hours=a['outer_hours'][0,6:])
    checks=dict(checkpoint_hashes=True,data_hash=True,source_protocol_hashes=True,stored_score_reproduction=True,
        grid_mean_query_parity=True,grid_random_query_parity=True,trained_encoder_causality=True,
        grid_random_chunk_consistency=True,full_grid_station_roundtrip_max_mm=roundtrip_error,
        marginal_preserving_shuffle=True,grid_sequence_shape=list(field.shape))
    write_json(ROOT/'verification.json',dict(checks=checks,diagnostics=diagnostics,
        bootstrap='5000 resamples of 12 monthly 24h blocks; equal-block means, exploratory',finalizer_sha256=sha256(__file__)))
    # Fixed highest-coverage station, not a best-performing illustration.
    p=all_pred[('17','ar')]; station=int(p['mask'].sum((0,1)).argmax()); block=6
    fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for axis,arm in zip(ax,['independent','ar']):
        p=all_pred[('17',arm)];s=p['samples'][block,:,:,station];y=p['target'][block,:,station];m=p['mask'][block,:,station]
        axis.plot(np.arange(24)[m],y[m],'k.-',label='Held-out GNSS')
        axis.plot(s[:8].T,color='#3079aa',alpha=.23)
        axis.fill_between(np.arange(24),*np.quantile(s,[.05,.95],axis=0),color='#3079aa',alpha=.15,label='90% interval')
        axis.set(title=arm,xlabel='Nominal hour',ylabel='ZWD (mm)');axis.legend()
    station_name=STORE.stations.iloc[int(p['station_ids'][station])]['station']
    fig.suptitle(f'{station_name}, 2018-07-16: same mean, different latent trajectories')
    fig.tight_layout();fig.savefig(ROOT/'sequence_comparison.png',dpi=150);plt.close(fig)
    # Scope-qualified report generated directly from measured records.
    lines=['# 连续生成场本机机制确认','',
      '2026-11-23。RTX4060真实数据训练；两种子各500步逐帧/GRU均值、350步独立/相关随机场。此轮为机制筛查，尚未验证最终GAN、Diffusion或VAE。',
      '', '输入仅A站Z0、温度、气压；从头训练。数据包历史PWV筛选限制仍保留，详见 ../20261123_sequence_spike_v01/RESEARCH.md。',
      '', '训练：2016—2017每月三个24小时片段，共72段；每段另有6小时热身。选择：2018每月1日。外层：2018每月16日，共12段、288小时，71个有效B外层站、19907站时。C/D标签未读取。B组及外层并非全新盲测。',
      '', '所有下表成绩采用统一144×96格网双线性查询和高度还原。CRPS为记录加权，RMSE为站宏。',
      '', '|种子|随机场|CRPS/mm|1h增量CRPS/mm|90%覆盖率|区间宽度/mm|', '|---|---|---:|---:|---:|---:|']
    for seed in ('42','17'):
        for arm in ('independent','ar'):
            v=r[seed][arm]['outer'];lines.append(f"|{seed}|{arm}|{v['crps']:.4f}|{v['increment_crps']:.4f}|{v['coverage90']:.3%}|{v['width90']:.3f}|")
    lines+=['','## 判断','']
    passed=True
    for seed in ('42','17'):
        i=r[seed]['independent']['outer'];v=r[seed]['ar']['outer']
        passed &= v['increment_crps']<i['increment_crps'] and v['crps']<=i['crps']*1.02
        ci=diagnostics[seed]['increment_crps_paired_blocks']['ci95']
        lines.append(f"- 种子{seed}：增量CRPS相对改善{(1-v['increment_crps']/i['increment_crps'])*100:.2f}%；等天气块差值95%区间[{ci[0]:.4f}, {ci[1]:.4f}] mm。独立噪声K128另见results.json。")
        sh=diagnostics[seed]['ar']['temporal_member_shuffle']['increment_crps']
        lines.append(f"  保持逐时全部边缘样本、打乱跨时成员对应后，增量CRPS由{v['increment_crps']:.4f}变为{sh:.4f} mm；此对照保持逐时空间联合场不变。")
        sp_i=diagnostics[seed]['independent']['original']['station_order_pair_crps'];sp_a=diagnostics[seed]['ar']['original']['station_order_pair_crps']
        lines.append(f"  固定站点编号相邻配对的空间差值CRPS：独立{sp_i:.4f}，相关{sp_a:.4f} mm；不是地理近邻或地形分层证据。")
    lines+=['',f'预登记小试门槛：{passed}。该门槛用于进入下一阶段机制验证，不等于最终论文方法通过。',
       '', '点精度与概率机制分开：两随机臂共享冻结均值，残差解析均值为零；集合均值RMSE受有限采样影响，不作为均值精度提升证据。']
    for seed in ('42','17'):
        lines.append(f"- 种子{seed}：逐帧均值站宏RMSE {r[seed]['frame']['outer']['macro_rmse']:.4f} mm；GRU均值 {r[seed]['gru']['outer']['macro_rmse']:.4f} mm。")
    lines+=['',f"匹配的50km高斯距离核背景RMSE {r['baseline']['macro_rmse']:.4f} mm；它不是旧IDW/粗双线性基线，不能与旧C144数字混用。",
       '', '## 尚待完成', '',
       '原始允许变量质控、真实海岸距离、地形分层与连续空间块检验、自监督编码消融、长于24小时的状态传递，以及正式三类生成模型均待后续验证。当前8维共享随机隐状态较简单，支持时间相关机制，不证明湿延迟细尺度物理恢复。',
       '', '本机已输出8成员×24小时×144×96的Z0序列，保留本地NPZ；图为固定高覆盖站的一段示例，不作整体性能证据。',
       '', '浏览器协作：指定内置浏览器后仍在工具初始化阶段失败；open_in_codex只返回queued，未观察到成功打开或模型回答。原始文献检索已完成，网页独立复核未完成。',
       '', f"成功运行耗时{r['runtime']['seconds']:.1f}秒（不含失败尝试和后验核查）；源码、配置、权重哈希与验证明细见同目录JSON。"]
    (ROOT/'RESULTS.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(checks=checks,passed=bool(passed),diagnostics=diagnostics)),flush=True)


if __name__=='__main__':main()
