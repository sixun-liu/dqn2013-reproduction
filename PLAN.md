# PLAN

> Updated: 2026-07-22T03:20:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：用冻结的 Nature 2015 DQN 基线，建立训练阶段、价值估计、内部表征、视觉关注和行为
  表现之间可复算的局部机制证据。
- Canonical baseline：`DQN2015-NATURE-BREAKOUT-INDEPENDENT`，EXP-0004 seed0、40 个 checkpoint、
  500 个固定 held-out states 和 40 次完整评估。
- 当前主要矛盾：已有高分策略和密集 checkpoint，但只有均值 Q 与 return 曲线；需要把固定状态、
  策略访问分布、训练 Q 语义和视觉干预拆开，避免把描述关联误写成因果。

## 阶段退出门

- [x] 六篇可解释性来源与 Claude 两份综合进入稳定 references，来源 hash 和本地方法边界已记录。
- [ ] 40 x 500 fixed-state Q/FC512 panel 通过 checkpoint、forward 和原 held-out-Q parity。
- [ ] 行为、匹配语义的 Q-return calibration、表征统计和视觉干预各有机器表与区间。
- [ ] 生成紧凑 DQN Statistical Atlas，观察、解释、反证和证据权限分开。
- [ ] 留下唯一下一因果问题，决定是否进入 Target Network 单变量实验。

## 活动路线

1. `EXP-0005` probe：先提取 fixed-state panel 和既有 behavior panel。
2. 通过 known-answer gate 后，采六阶段 calibration trajectories。
3. 对固定采样做四帧消融和 Gaussian-blur saliency，配随机/全局扰动控制。
4. 计算 effective rank、linear CKA、PCA；t-SNE 只作人工探索图。
5. 汇总价值、表征、视觉敏感性和行为的阶段关系，结案后再裁决训练消融。

## Parked Lanes

- 2013 checkpoint 对比：网络、预处理、replay 和优化器混杂，只作历史附录。
- Target Network、Double DQN、replay 和 frame-stack 训练消融：等待 EXP-0005 结案。
- AtariARI 依赖与 RAM linear probe：现有分析不能回答表征语义时再接入。
- Nature 多 seed、其他 Atari 和精确历史 ALE/Torch7 重建。
