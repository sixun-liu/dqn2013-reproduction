# CURRENT_STATE

> Updated: 2026-07-20T16:18:38Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

2013 独立重实现已经给出学习趋势和单位修正的稳定性证据，但其代码谱系与历史协议风险不适合继续
承担主要复现预算。用户已决定把 Nature 2015 Breakout 升为当前候选，2013 长程扩展停止在现有
`promising_unresolved` 证据处。

## 当前主要矛盾

DeepMind DQN 3.0 提供 Nature 2015 作者代码和较强协议锚点，但 Lua/Torch7/Xitari 栈不适合作为
当前 GPU 的直接执行起点；CleanRL 可现代运行，却在优化器、预算、epsilon、target 更新和 wrapper
等方面偏离作者配置。正式计算前必须先冻结论文主张、执行实现、ALE 协议与成本包络。

## 下一项决策

完成 Nature 2015 的 claim-protocol 对账并选择一个许可清楚、协议可配置的现代 executor；随后只跑
最小 smoke/pilot 实测 ETA，由用户确认成本后再冻结首个 replication。2013 的两项 pending human
review 继续保留，但不再作为 Nature 协议恢复的阻塞项。
