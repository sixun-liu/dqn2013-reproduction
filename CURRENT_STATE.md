# CURRENT_STATE

> Updated: 2026-07-22T04:50:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0005 的固定状态图谱有效；EXP-0006/0007 的完整游戏 calibration 因跨 checkpoint 的纯
greedy 游戏长度差异两次未达共同完成门，Q-G 问题保持 unknown，部分 trace 全部排除。

## 当前主要矛盾

无 sticky action 的纯 greedy Breakout 完全确定，需要显式初始状态集合；EpisodicLife 只定义
训练 terminal，不缩短真实完整游戏。校准若重开必须在固定 decisions 和固定 games 中二选一，
当前不再重试，也不把该工程失败解释成 9.25M 方法正负证据。

## 下一项决策

直接采用 Greydanus 作者公式，在 `9.0M / 9.25M / 9.5M / 10M` 的同一固定状态子集上做
四帧消融、局部 blur、随机 cell 与全局 blur 控制。输出只支持局部输入敏感性；Atrey 仓因无
license 文件与 Toybox 依赖只读参考，当前不启动新训练。
