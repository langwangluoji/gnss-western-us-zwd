# Europe VMF3 Background Implementation Plan

> **For agentic workers:** Use executing-plans to implement this plan inline. Verify each task before the next; one independent final review.

**Goal:** 引入同源VMF3背景特征，实测其对冻结欧洲最佳场的额外点RMSE收益。

**Architecture:** 冻结p14＋GRU＋I1，16步逐查询VMF3三特征经GRU32，拼接既有111维条件，通过零初始化MLP校正场。匹配无VMF3校正器，另重训CNN＋VMF3。

**Tech Stack:** 本机Anaconda pytorch、NumPy、h5py、原Model4 VMF3模块、XGBoost、Matplotlib。

**Spec:** [冻结合同](../../../research/europe_vmf3_20261126/README.md)

## Global Constraints

- 继承273/64/63/11物理站址分组及2020–2023/2024边界；不覆盖旧结果。
- 新模型VMF3外的输入不变，目标站GNSS/T/P不进入条件；P对齐由已有参考站推导。
- VMF3缺失不从未来有效时刻填充；16记录为90h跨度。
- 两臂同种子、3000连续块、相同布局；2023开发选权重，2024回看。
- 数据/权重/数组留本地，轻量证据核验后推送。

## Review Focus

- VMF3米与标签mm一致；逐格点垂直后水平，坐标经度跨零正确。
- 新模型aligned不暗中读取目标站气象或湿度，anomaly减本模型背景。
- 缺历元、训练/推理mask和历史年龄一致；任何条件不晚于预测起点。
- 校正头初始严格保持旧模型，control/vmf3预算及样本相同。
- 同键比较、集成定义、VMF3原生背景基线与CNN新对照来源清晰。

## Tasks

- [ ] P0：VMF3插值、源SHA、缺测与单位；手算单位/高程传递/缺历元回归先失败再实现。
- [ ] P1：背景三特征与数据隔离测试；GRU校正器零初始化和真实CUDA四更新。
- [ ] P2：双种子control/vmf3各3000连续块；CNN＋VMF3按原配置双种子重训。
- [ ] P3：封存2023开发选择，2024固定键RMSE、季节/重点站与背景单独基线评价。
- [ ] P4：独立复算/审阅、渲染图、完整报告；提交推送并核对远端HEAD。
