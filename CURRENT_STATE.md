# CURRENT_STATE

> Updated: 2026-07-17T05:26:23Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0003` 已自然完成并通过预注册支持门槛：只修正 replay/epsilon agent-step 单位后，首次
退化窗口的 Q mean max 从 61.74 降至 3.72，配对终点评估从 2.51 提高到 10.04。单位缩短是
EXP-0001 首次失稳的重要促成因素，但单 seed 结果不构成论文 168 分的数值复现。

## 当前主要矛盾

已排除“最终低分只是单一评估 seed 偶然”，并取得 replay/epsilon 联合单位修正的正向干预证据。
剩余差距是修正后的稳定性尚未覆盖 Figure 2 约 5M updates，且现代 ALE/独立重写与历史协议仍有漂移。

## 下一项决策

先由用户复核 `artifacts/dqn2013/review/LATEST.md`。若继续投入 DQN 计算，下一正式候选应从头运行
协议对齐的约 5M-update 单 seed 基线，检验长期稳定与论文尺度；在这项证据前不测试
NIPS optimizer/gamma，也不加入 2015 target network。
