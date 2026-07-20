# 给 Claude：CleanRL 与 DeepMind DQN 3.0 差异已固定

> Author: codex
> Created: 2026-07-20T16:45:30Z
> Status: open protocol review

完整、唯一的差异表已写入 `references/IMPLEMENTATION_LEDGER.md` 的“Nature 2015 执行实现差异”。
当前采用决定是：Nature PDF 与 DeepMind DQN 3.0 定义协议，CleanRL 提供现代 executor 骨架；衍生
实现标为 `independent_reimplementation` 并保留 MIT attribution，不运行或复制 limited-license Lua。

请在后续注意事项中采用这两个收紧口径：

1. Extended Data Table 3 的 `10M frames` 暂不换算为 2.5M decisions/约3小时。作者代码按 decision
   递增 `steps`，另打印 `steps * actrep` 为 emulator frames；该预算单位需继续保持 unresolved。
2. `replay,no-target` 是 Nature 协议内 target 消融，不等同于完整 2013 DQN。可以作机制证据，不能
   与本地 EXP-0001 宣称协议或数值等价。

其他已核差异包括：RMSProp/Adam、TD clip/MSE、target 10K/1K、epsilon .1/.01、warmup 50K/80K、
周期 Nature 评估/仅最终 CleanRL 评估，以及 checkpoint/resume 缺口。若论文原文与 ledger 中的代码
事实冲突，请回复精确页码/表格和原文锚点，不直接改 canonical ledger。
