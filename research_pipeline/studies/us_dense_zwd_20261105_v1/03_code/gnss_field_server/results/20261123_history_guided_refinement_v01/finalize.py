from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent
r=json.loads((R/'confirmation_v02/results.json').read_text(encoding='utf-8'))
p=np.load(R/'confirmation_v02/predictions.npz')
y=p['truth'];a=p['wide_42'];b=p['wide_17_original']
def score(v,m):
 n=m.sum(0); ok=n>0
 return float(np.sqrt(np.where(m,(v-y)**2,0).sum(0)[ok]/n[ok]).mean())
months=p['hours'].astype('datetime64[h]').astype('datetime64[M]');groups=[np.flatnonzero(months==v) for v in np.unique(months)]
def delta(idx,mask):
 m=mask[idx];n=m.sum(0);ok=n>0
 return float((np.sqrt(np.where(m,(a[idx]-y[idx])**2,0).sum(0)[ok]/n[ok])-np.sqrt(np.where(m,(b[idx]-y[idx])**2,0).sum(0)[ok]/n[ok])).mean())
output={}
for key in ('mask','common_mask'):
 m=p[key].astype(bool);rng=np.random.default_rng(42)
 draws=[delta(np.concatenate([groups[j] for j in rng.integers(0,12,12)]),m) for _ in range(2000)]
 output[key]={'difference':score(a,m)-score(b,m),'ci95':np.quantile(draws,[.025,.975]).tolist(),'monthly_difference':[delta(g,m) for g in groups]}
(R/'champion_comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
fig,ax=plt.subplots(figsize=(8,4.5));names=['small_42','small_17','wide_42','wide_17'];x=np.arange(4)
ax.bar(x-.18,[r['models'][n]['original']['macro_rmse'] for n in names],.36,label='Frozen U-Net',color='#7896ad')
ax.bar(x+.18,[r['models'][n]['corrected']['macro_rmse'] for n in names],.36,label='With A-innovation readout',color='#228877')
ax.set_xticks(x,names);ax.set_ylabel('Station-macro RMSE (mm)');ax.set_ylim(8,10);ax.legend();ax.set_title('2019 development replay: 144 hours, 79 stations')
fig.tight_layout();fig.savefig(R/'local_confirmation.png',dpi=180);plt.close(fig)
rows=['|Frozen source|Original|Corrected|Paired difference 95% CI|','|---|---:|---:|---|']
for n in names:
 d=r['models'][n];rows.append(f"|{n}|{d['original']['macro_rmse']:.4f}|{d['corrected']['macro_rmse']:.4f}|{d['paired_ci'][0]:.4f}, {d['paired_ci'][1]:.4f}|")
control=json.loads((R/'objective_probe/results.json').read_text(encoding='utf-8'))
ctrl=['|Source|Constant|Amplitude|Three horizontal kernels|Selected geometry|','|---|---:|---:|---:|---:|']
outer=[9.55316,9.46379,9.63180,9.69408]
for n,v in zip(names,outer):
 c=control[n]['models'];ctrl.append(f"|{n}|{c['constant']['outer']['macro_rmse']:.4f}|{c['amplitude']['outer']['macro_rmse']:.4f}|{c['geo3_no_height']['outer']['macro_rmse']:.4f}|{v:.4f}|")
text='''# 本机验证结论：冻结气象 U-Net 的 A 站创新校正

本轮找到可继续验证的精度改进流程。四个冻结起点各自在 B 内层选择低维读出，随后在相同 C144 开发回看上均改善；这并不等于同一个固定配置已跨种子复现。现有最佳产品暂保留。

## 模型与训练
输入为四帧 A 站 IDW 五变量、有效掩码、GNSS splat、地形坐标和周期信息，共42通道。气象量为 A 站位 ERA5 采样，不是完整独立气象格网。冻结 U-Net 在 A 站的公共高度预测与真实 A 观测作差；按50/100/200 km及不限制/750 m高差形成6个核，构造核均值或24维均值、方向、高差矩。以 B357 训练站、863合格训练小时拟合 FP64 站均衡 MSE 岭回归，内层留站留天气选择配置。输出完整144×96格网，经统一双线性查询和高程恢复评分。小网络16/32/64、124529参数；宽编码器32/64/128、375473参数。

## 扩展开发回看
每月8—10日每6小时，共144小时、79站、11050有效站时，单位mm。

'''+ '\n'.join(rows)+'''

区间来自2000次同步月份天气块重采样，共12块，不包含历史选型不确定性。C2019已有历史开发暴露；本轮日期与原48小时不同，不称盲测。D2020未读取。此前 B 外层三组区间低于零，小网络seed42区间跨零。

## 是否替换原最佳产品
新候选wide42由内层选出，旧优选wide17同样由原内层选出，保持此选择。原生79站上新减旧的站宏RMSE差值为 %.5f mm，95%%区间[%.5f, %.5f]。共同76站、10610站时差值为 %.5f，区间[%.5f, %.5f]。新候选微RMSE为9.62941，旧优选为9.50843 mm。因此本轮支持校正流程有用，尚不支持全面替换产品。

## 简单控制：B外层站宏RMSE
'''%(output['mask']['difference'],*output['mask']['ci95'],output['common_mask']['difference'],*output['common_mask']['ci95'])+'\n'.join(ctrl)+'''

常数和振幅再校准未解释主要收益。仅水平三核弱于所选几何方案，但24维方向矩的独立贡献仍需单因素检验，不能把整块收益均归给方向矩。直接平滑站宏RMSE的凸读出改善内层、外层有升有降，保留原锁定岭回归候选。仅重估深层特征读出收益小且不一致，当前证据不足以将瓶颈归于末层优化不足。

## 工程验收
48个全格网案例的发布查询最大误差小于0.000053 mm，低于0.001 mm阈值。首次TF32路径最大差约0.0018 mm，失败记录保留；最终扩展指标与发布验证采用禁用TF32的严格FP32。原冻结权重和已拟合系数未改动。初始拟合特征来自原TF32路径。详细锁文件、脚本、预测和失败记录均保留。本轮没有提交服务器作业。

## 下一步
先锁定当前新旧两候选、共同站群、有效站时和主指标，在本机完成完整2019开发年配对评价，报告站宏/微RMSE、季节稳定性及同步7日块bootstrap。当前收益来自确定性创新读出，不归于GAN、Diffusion或VAE；概率模型的改进须在固定均值后独立检验CRPS/ES和校准。25 km仍仅为输出间距，本轮不主张已恢复真实细尺度结构。全年证据支持之后再安排服务器扩大训练。
'''
(R/'RESULTS.md').write_text(text,encoding='utf-8')
(R/'PRO_REVIEW.md').write_text('''# Pro协作记录
会话：https://chatgpt.com/c/6b03decc-5704-83ee-bc10-518a7563aa26

两轮回复均已完整读取。首轮建议先诊断冻结读出，再验证A创新几何增量，按修正方向与幅度决定是否受限微调。特别区分站均衡MSE与站宏RMSE。
第二轮判断本轮可按“小试可行、保留旧最佳产品”收尾，无须新增训练；要求补算锁定新旧冠军配对区间、逐月差值以及常数/振幅/无高差控制，并限定跨种子复现的表述。上述统计见champion_comparison.json与RESULTS.md。未依据Pro意见改变已锁候选。建议属于研究评议，数值结论来自本地脚本。
''',encoding='utf-8')
print(json.dumps(output,indent=2))
