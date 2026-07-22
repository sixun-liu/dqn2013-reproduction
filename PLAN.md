# PLAN

> Updated: 2026-07-22T03:56:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：用冻结的 Nature 2015 DQN 基线，建立训练阶段、价值估计、内部表征、视觉关注和行为
  表现之间可复算的局部机制证据。
- Canonical baseline：`DQN2015-NATURE-BREAKOUT-INDEPENDENT`，EXP-0004 seed0、40 个 checkpoint、
  500 个固定 held-out states 和 40 次完整评估。
- 当前主要矛盾：EXP-0005 已发现 `9.25M` 的高 Q、低 margin、低回报局部异常；需要用匹配训练
  语义的轨迹区分价值校准、策略访问分布和评估方差，再决定是否投入视觉干预。

## 阶段退出门

- [x] 六篇可解释性来源与 Claude 两份综合进入稳定 references，来源 hash 和本地方法边界已记录。
- [x] 40 x 500 fixed-state Q/FC512 panel 通过 checkpoint、forward 和原 held-out-Q parity。
- [ ] 行为、匹配语义的 Q-return calibration、表征统计和视觉干预各有机器表与区间。
- [ ] 生成紧凑 DQN Statistical Atlas，观察、解释、反证和证据权限分开。
- [ ] 留下唯一下一因果问题，决定是否进入 Target Network 单变量实验。

## 活动路线

1. `EXP-0005` 已关闭：fixed-state panel、行为表、表征谱、CKA 和 block bootstrap 已完成。
2. 下一 probe 比较 `9.0M / 9.25M / 9.5M / 10M` 的匹配语义 calibration trajectories。
3. calibration 若确认局部异常，再对相同阶段做四帧消融和 Gaussian-blur saliency，配随机/全局
   扰动控制；若不确认，先按评估方差或状态覆盖解释，不扩大视觉实验。
4. 汇总价值、表征、视觉敏感性和行为的阶段关系，结案后再裁决训练消融。

## Parked Lanes

- 2013 checkpoint 对比：网络、预处理、replay 和优化器混杂，只作历史附录。
- Target Network、Double DQN、replay 和 frame-stack 训练消融：等待 calibration/视觉证据闭环。
- AtariARI 依赖与 RAM linear probe：现有分析不能回答表征语义时再接入。
- Nature 多 seed、其他 Atari 和精确历史 ALE/Torch7 重建。
