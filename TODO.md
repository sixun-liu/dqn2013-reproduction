# TODO

> Updated: 2026-07-22T05:36:29Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期未完成动作；实验事实和完成历史不堆在这里。

## Now

- [ ] [user] 复核四阶段局部机制综合；trigger: 打开 `ART-0032`，决定是否继续 DQN 机制路线。
- [ ] [user] 复核 EXP-0005 固定状态图谱；trigger: 打开 `ART-0028`，不阻塞下一 diagnostic probe。
- [ ] [user] 复核 EXP-0004 既有主图；trigger: 打开 `ART-0019`，不阻塞 diagnostic probe。
- [ ] [user] 复核 EXP-0008 视觉干预图；trigger: 打开 `ART-0031`，不阻塞跨周期综合。

## Waiting

- [ ] [codex] Margin 匹配的 state-level switch 分析；trigger: 用户复核后仍把低 margin 列为首要缺口。
- [ ] [codex] AtariARI RAM/object-region probe；trigger: 跨周期综合后视觉语义仍是首要缺口。
- [ ] [codex] Target Network 因果消融；trigger: 跨周期综合形成可区分的单变量假说。
