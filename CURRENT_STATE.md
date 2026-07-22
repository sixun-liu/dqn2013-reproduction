# CURRENT_STATE

> Updated: 2026-07-22T03:56:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0005 已把 EXP-0004 的 40 checkpoint 与同一 500 状态恢复成可复算 Q/FC512 图谱：Q 绝对
尺度不是策略质量的充分代理，`9.25M` 是下一轮最有信息量的局部异常。

## 当前主要矛盾

`9.25M` 的 mean max-Q 全程最高 `5.315`，行为均分却只有 `141.0`；`10M` 在 mean max-Q
降到 `3.671` 时达到 `350.18`，同时 action margin 达全程最高。原始阶段相关经一阶差分后大幅
减弱，说明共同训练进度、局部校准和策略访问分布仍混杂。

## 下一项决策

比较 `9.0M / 9.25M / 9.5M / 10M` 的 greedy calibration trajectories，严格匹配 clipped
reward、life-loss terminal 和 `gamma=0.99`。只有轨迹证据确认局部异常，才进入四帧消融与
带随机/全局控制的 saliency；当前不启动新训练。
