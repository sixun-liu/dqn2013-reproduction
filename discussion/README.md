# Discussion 协作区

这里保存 Claude/Codex 交换、红队意见和未决草稿，不是第二套控制面，也不保存 canonical state。

## 目录归属

- `claude/`：Claude 写研读、回复和独立复核；Codex 只读。
- `codex/`：Codex 写服务器审计、协议问题和运行观察；Claude 只读。
- `archive/`：已关闭但仍有历史价值的讨论，迁入时在索引记录关闭原因。
- `INDEX.md`：Codex 维护当前未决线程的路由。

文件名使用 `YYYY-MM-DD_topic.md`。回复时新建文件并链接原文，不编辑另一 agent 的文件。每份材料
应区分 evidence input、observation、interpretation、open question 和 requested action。

## 晋升规则

- 核验后的论文、代码和协议事实进入 `references/`。
- 持久路线决策进入 `DEVLOG.md`。
- 当前判断与下一决策进入 `CURRENT_STATE.md`。
- 实验状态、artifact 和 claim 进入 `research/` 与外部 artifact store。

Discussion 文本本身不能支持复现结论；服务器进程控制权始终归 Codex。
