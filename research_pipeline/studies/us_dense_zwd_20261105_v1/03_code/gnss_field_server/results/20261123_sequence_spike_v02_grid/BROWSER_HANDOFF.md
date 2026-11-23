# 待网页端独立复核的材料

状态：未发送。用户指定内置浏览器；cua在初始化/创建iab页面阶段均报“failed to write kernel assets: 系统找不到指定的路径”。桌面open_in_codex返回queued，未观察到打开成功或网页回答。无需重复上传整个项目。

请审阅公开项目 https://github.com/langwangluoji/gnss-western-us-zwd 中当前工程下本目录RESULTS.md、PROTOCOL.md、verification.json，以及相邻20261123_sequence_spike_v01/RESEARCH.md和tools/sequence_mechanism_spike.py、sequence_grid_spike.py。

目标是稀疏GNSS和站点气象条件下、考虑地形的连续ZWD场序列，拟形成方法创新论文。允许监督、自监督、无监督探索；禁止PWV/IWV/TCWV条件和教师，现阶段仅A站Z0/T/P，无独立稠密真值。所有可用研究数据已在现有项目，不能假定有HRRR式训练真值。

实测：统一格网查询下，两个种子时间相关隐状态相对独立噪声改善1h增量CRPS；逐时保留全部样本但打乱成员时间对应后，增量评分变差。GRU均值反而弱于逐帧MLP，故不能把收益归于更强时间编码器。当前为8维线性+中心化二次潜变量生成场，不是最终Diffusion/GAN/VAE。旧队列以PWV完整性筛选过，已披露，须从原始文件重建允许变量质控。

请重点提出反例和可检验替代解释：
1. 此时间机制证据能支持哪些主张，哪些不能？
2. 没有稠密真值时，哪种空间监督/自监督编码最能支持地形细尺度变化？
3. 若迁移到Diffusion、VAE、GAN，应优先比较什么，并怎样公平匹配信息条件和预算？
4. 是否存在更简单的概率基线可解释全部收益？
5. 给出原始论文依据和低成本配对试验，明确理论建议与本项目已实测证据。

不得将自己的建议或第二模型意见称为独立实验验证；不假设访问本地未上传权重。客户端将核对原始来源和真实实验，再决定采用、修正或拒绝。
