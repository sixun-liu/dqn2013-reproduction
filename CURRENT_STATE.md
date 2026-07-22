# CURRENT_STATE

> Updated: 2026-07-22T05:28:49Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0008 的作者方法视觉干预有效，但否定“9.25M 更广泛视觉脆弱”：该阶段的局部/全局/帧扰动
幅度没有更高，真正突出的局部现象是更低 action margin 与更高空间集中度。

## 当前主要矛盾

已有证据分散在三个周期：EXP-0004 提供行为，EXP-0005 提供 Q/FC512，EXP-0008 提供 margin、
视觉集中度和干预敏感性。Q-return calibration 仍为 unknown；mean(score/Q-energy) 又暴露近零
分母尾部，因此下一步先做跨周期综合，不立即扩展新 probe。

## 下一项决策

把 `9.0M / 9.25M / 9.5M / 10M` 的行为、Q、FC512、margin 与视觉机器表合并成一张局部证据图；
只有综合后低 margin/高集中仍是首要缺口，才在 margin 匹配分析与 AtariARI/RAM 对象区域 probe
之间选择一个预注册问题。Atrey 继续只读，不启动新训练。
