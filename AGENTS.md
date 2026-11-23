# 当前项目范围与验证

- 当前研究：美国西部25–50°N、125–110°W，2016–2020，A174/B511/C110/D110。先读 `docs/CURRENT_RESEARCH.md` 和 `docs/EXPERIMENT_INDEX.md`。
- 当前工程：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server`。旧39站/2008–2011结果不能替代当前扩区域结果。
- 回答“最新”之前检查工程 `results/` 的实际目录、状态、实验锁与数据角色，不能只依赖外层README或历史运行状态。
- 按工程内 `AGENTS.md` 和 `docs/METHOD_VALIDATION.md`，方法可行性需要真实环境验证；区分本机短训、开发回看、完整年度和服务器规模。C2019已开发暴露；D2020读取情况按实验锁核查。
- 目录维护不得静默改写科学锁、自动重训或提交收费服务器任务。原始数据、唯一权重和负结果本地保留；Git只同步代码与轻量证据，凭据不提交。
- 历史归档在项目外 `../GAN对流层_历史归档_20261123`。当前上游依赖保留原路径；再次清理先读 `maintenance/cleanup_20261123/README.md`。

## 写作

沿用用户全局写作偏好：中文anti-defensive-writing、英文对应英文技能；研究文字准确保留事实和证据范围。文章任务共同应用nature-writing、nature-polishing、humanizer-zh-academic及stss，DOCX另用documents；原文修改以完整原始文件为依据。

## GitHub持续同步（2026-11-23用户要求）

- 本项目后续每次产生代码、配置、文档或实验进展变更，完成相应核验后，默认提交并推送至公开仓库 `https://github.com/langwangluoji/gnss-western-us-zwd`，无需逐次请求同步许可。
- 每次同步核对本地与远端提交一致。同步失败时明确说明，保留本地提交等待恢复；不将未推送内容称为已同步。
- 沿用当前轻量同步边界：原始数据、权重、大型数组、缓存及凭据保留本地；实验进展同步协议、指标、报告和来源/哈希清单。只读查询没有文件变更时无需创建空提交。
