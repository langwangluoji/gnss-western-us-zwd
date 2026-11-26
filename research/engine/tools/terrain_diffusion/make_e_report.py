import json
from pathlib import Path
import matplotlib.pyplot as plt
rows=json.loads(Path('results/20261124_E_summary.json').read_text())
labels=[f"{r['experiment']}-{r['sampling']}" for r in rows]; vals=[r['mean'] for r in rows]
plt.figure(figsize=(10,4)); plt.bar(labels,vals,color=['#4472c4' if r['sampling']=='random' else '#ed7d31' for r in rows]); plt.axhline(10.806,color='black',ls='--',label='p14 10.806 mm'); plt.ylabel('Formal macro RMSE (mm)'); plt.xticks(rotation=45); plt.legend(); plt.tight_layout(); plt.savefig('results/20261124_E_summary.png',dpi=180); plt.close()
report='''# E01–E05 两小时短训评价\n\n本轮按统一 p14 formal 评价合同运行 E01–E05，每个实验 random/hole100、seed42/17、60 秒续训。结果用于工程筛查。当前五个实验入口已锁定机制开关，但短训调用的是现有 P15 固定支撑训练器，因此这些数值是合同对照，不能解释为地形编码、边消息、GRU、增量损失或残差扩散已经生效。\n\n| 实验 | random 均值 | hole100 均值 |\n|---|---:|---:|\n'''
for e in ['E01','E02','E03','E04','E05']:
 a=next(r['mean'] for r in rows if r['experiment']==e and r['sampling']=='random'); b=next(r['mean'] for r in rows if r['experiment']==e and r['sampling']=='hole100'); report+=f'| {e} | {a:.3f} | {b:.3f} |\\n'
report+='''\n正式参考：p14 均值 10.806 mm；P15-hole100 10.804 mm。短训结果约 11.4–11.6 mm，说明 60 秒不足以判断结构优劣。下一步应把机制开关真正接入模型前向和损失，再按同一矩阵进行 600 秒续训。训练日志与原始 formal_summary.json 保留在各实验目录。\n'''
Path('results/20261124_E_summary_REPORT.md').write_text(report,encoding='utf-8')
