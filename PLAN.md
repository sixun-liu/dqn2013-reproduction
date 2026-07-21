# PLAN

> Updated: 2026-07-21T02:34:37Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `reproduction`
- 北极星：理解 DQN 从 2013 到 Nature 2015 的稳定性演进，并在现代可审计环境中跑通 Nature
  2015 Breakout 基线。
- 当前主要矛盾：单 seed 已复现论文分数量级，但现代 ALE 与固定学习率协议不能支持严格等价或
  跨 seed 稳定性主张；DQN 作为 DreamerV3 前置铺路不再追加当前预算。

## 阶段退出门

- [x] 固定 Nature 2015 论文 PDF、版本、hash 和 Extended Data Table 3 Breakout 目标主张。
- [x] 对账 2013 paper、本地实现、DeepMind DQN 3.0 和现代 executor 的谱系与协议。
- [x] 冻结 ALE/wrapper、预算单位、评估协议、参考产物和接受包络。
- [x] 通过 pilot 成本门并完成 10M-decision replication；结果与限制由 `EVT-0032` 关闭。

## 活动路线

1. 保留 `EXP-0004` 的 raw run、checkpoint、同坐标图和协议差异表，等待用户人工图审。
2. 与 DreamerV3 walker 结果合并为双论文中文总结和导师日报素材。
3. 不启动新 DQN 计算；只有用户要求提升证据权限时才为第二固定协议 seed 新建 replication cycle。

## Parked Lanes

- 2013 独立重实现约 5M-update 长程扩展。
- 只给 2013 实现加入 target network 的单变量机制实验。
- 多训练 seed、其他 Atari 游戏和精确历史 ALE/Torch7 环境重建。
- 第二固定协议 seed；这是提升当前单 seed 主张的下一判别问题，但未排期。
