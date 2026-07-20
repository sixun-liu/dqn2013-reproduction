# Discussion Index

> Updated: 2026-07-20T16:45:30Z
> Maintainer: codex
> Source of truth: manual routing index

## Open Threads

1. **Nature 2015 Breakout 主张与协议恢复**
   - Codex 交接：`codex/2026-07-21_nature2015-route-and-provenance.md`
   - Executor 预检：`codex/2026-07-21_nature2015-executor-preflight.md`
   - CleanRL/作者代码差异回复：`codex/2026-07-21_cleanrl-vs-deepmind-reply.md`
   - Claude 协议回复：`claude/2026-07-21_nature2015-protocol-reply.md`
   - Claude 收紧确认：`claude/2026-07-21_cleanrl-vs-deepmind-claude-ack.md`
   - Claude 性价比计划：`claude/2026-07-21_dqn-experiment-plan.md`
   - Codex 下一动作：将抽验事实写入 canonical protocol audit 并冻结 executor。
   - 关闭条件：核验事实进入 `references/`，首个 reproduction target 和成本包络可冻结。

2. **现代 executor 与许可边界**
   - 已知：DeepMind DQN 3.0 是 limited academic review author code；CleanRL 是 MIT third party。
   - 当前决定：P2/A1 定义协议，T2 提供现代执行骨架；衍生 executor 保留 MIT attribution。
   - 关闭条件：lineage、license、协议差异和运行入口均写入 implementation ledger。

## Closed Threads

1. **是否继续 2013 独立实现长程扩展**
   - 用户于 2026-07-21 本地时间决定停止扩展；EXP-0001--EXP-0003 保留为历史与机制证据，
     Nature 2015 升为主线。
