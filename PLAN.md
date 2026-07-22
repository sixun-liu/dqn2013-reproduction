# PLAN

> Updated: 2026-07-22T04:50:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：用冻结的 Nature 2015 DQN 基线，建立训练阶段、价值估计、内部表征、视觉关注和行为
  表现之间可复算的局部机制证据。
- Canonical baseline：`DQN2015-NATURE-BREAKOUT-INDEPENDENT`，EXP-0004 seed0、40 个 checkpoint、
  500 个固定 held-out states 和 40 次完整评估。
- 当前主要矛盾：EXP-0006/0007 已证明固定完整游戏数的纯 greedy calibration 在当前预算下不可比，
  Q-G 问题保持 unknown；需要用固定 500 状态的受控输入干预回答更窄的视觉敏感性问题。

## 阶段退出门

- [x] 六篇可解释性来源与 Claude 两份综合进入稳定 references，来源 hash 和本地方法边界已记录。
- [x] 40 x 500 fixed-state Q/FC512 panel 通过 checkpoint、forward 和原 held-out-Q parity。
- [x] calibration 路线以完成门失败和明确 unknown 关闭；不得用部分 trace 补结论。
- [ ] 表征统计与视觉干预各有机器表、配对区间和正负控制。
- [ ] 生成紧凑 DQN Statistical Atlas，观察、解释、反证和证据权限分开。
- [ ] 留下唯一下一因果问题，决定是否进入 Target Network 单变量实验。

## 活动路线

1. `EXP-0005` 已关闭：fixed-state panel、行为表、表征谱、CKA 和 block bootstrap 已完成。
2. `EXP-0006/0007` 已关闭：纯 greedy 完整游戏成本跨 checkpoint 差异过大，Q-G 未裁决。
3. 下一 probe 在相同 128/500 固定状态上做四帧消融与 Greydanus Gaussian-blur saliency，配
   随机 cell、全局 blur 和 fixed-state paired bootstrap。
4. saliency 只生成可证伪的局部输入敏感性假说；没有自然反事实时不解释成对象因果。
5. 汇总价值、表征、视觉敏感性和行为，结案后再决定是否需要 AtariARI 或训练消融。

## Parked Lanes

- 2013 checkpoint 对比：网络、预处理、replay 和优化器混杂，只作历史附录。
- Target Network、Double DQN、replay 和 frame-stack 训练消融：等待 calibration/视觉证据闭环。
- AtariARI 依赖与 RAM linear probe：现有分析不能回答表征语义时再接入。
- Nature 多 seed、其他 Atari 和精确历史 ALE/Torch7 重建。
