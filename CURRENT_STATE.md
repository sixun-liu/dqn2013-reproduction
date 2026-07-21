# CURRENT_STATE

> Updated: 2026-07-21T02:34:37Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

Nature 2015 Breakout `EXP-0004` 已在冻结的现代独立重实现上完成 10M agent decisions；40 次周期
评估 peak/final mean 均为 350.18，相对论文 Table 3 的 316.8 高 10.54%。本轮复现了论文分数量级，
但单 seed、现代 ALE 和固定单一学习率使裁决保持 `promising_unresolved`。

## 当前主要矛盾

DQN 的当前目标是给 DreamerV3 复现铺设 RL 基础，而不是提升为多 seed、全 Atari 论文复现。
现有 checkpoint 已独立复评且逐 episode 完全一致，继续增加 seed 的边际收益低于转回 DreamerV3
交付整合。剩余不确定性应保留为边界，不能用单条高分曲线外推跨 seed 稳定性。

## 下一项决策

停止新增 DQN 计算，把 `EXP-0004` 对照图交用户复核，并与既有 DreamerV3 walker 结果合并成中文
双论文复现总结和日报素材。若未来需要提升 DQN 证据权限，唯一下一判别问题是第二固定协议 seed
能否仍进入 316.8 量级；该问题目前明确 parked。
