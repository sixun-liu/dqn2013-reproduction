# CURRENT_STATE

> Updated: 2026-07-22T05:36:29Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

固定 Nature DQN baseline 的局部机制证据已综合为 `ART-0032` / `C-0001`：9.25M 行为低谷与
高 max-Q、低 margin、FC512 回撤、低扰动幅度和高空间集中度共现，不支持广泛视觉脆弱解释。

## 当前主要矛盾

该结论仍是单 seed、固定早期 replay 面板上的 `supported-working-claim`，不是训练稳定性、对象语义
或行为因果。Q-return calibration 仍为 unknown；mean(score/Q-energy) 的近零分母尾部已作为
反证写入 claim，不能再把该均值当稳健跨阶段标尺。

## 下一项决策

用户先复核 `ART-0032` 综合图与 `ART-0031` 显著图；若继续 DQN 机制路线，在 margin 匹配分析、
AtariARI/RAM 对象区域 probe 和 Target Network 单变量训练消融中只选择一个新问题预注册。
Atrey 继续只读，未做路线选择前不启动新训练。
