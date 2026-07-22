# CURRENT_STATE

> Updated: 2026-07-22T03:20:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

Nature 2015 Breakout EXP-0004 已完成单 seed 分数量级复现，并留下 40 个周期 checkpoint、500 个
固定状态和 40 次完整评估；当前从 `reproduction` 进入 `exploration`，用这些冻结产物建立 DQN
价值、表征、视觉敏感性和行为随训练变化的局部证据。

## 当前主要矛盾

现有 mean max-Q 和 return 曲线不能区分状态分布变化、价值校准、表征几何与视觉输入敏感性。
EXP-0005 将固定状态面板与阶段策略轨迹分栏，并统一 reward/terminal/discount 语义。所有阶段关联
保持描述性，saliency 只在正负控制通过后解释，2013/2015 差异不用于 Target Network 归因。

## 下一项决策

先完成 40 x 500 fixed-state Q/FC512 panel 的 known-answer 和原 held-out-Q parity。只有该门通过，
才采 calibration trajectory 和视觉干预；EXP-0005 结案前不启动新训练。
