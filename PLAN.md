# PLAN

> Updated: 2026-07-20T16:18:38Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `understanding`
- 北极星：理解 DQN 从 2013 到 Nature 2015 的稳定性演进，并在现代可审计环境中跑通 Nature
  2015 Breakout 基线。
- 当前主要矛盾：作者代码可恢复论文配置，但现代 executor 与历史 ALE/评估协议尚未闭合。

## 阶段退出门

- [ ] 固定 Nature 2015 论文 PDF、DOI、版本、hash 和目标 Breakout 结果主张。
- [ ] 对账 2013 paper、本地实现、DeepMind DQN 3.0 和现代 executor 的谱系与协议。
- [ ] 冻结 ALE/wrapper、预算单位、评估协议、参考产物和接受包络。
- [ ] 用最小 smoke/pilot 实测吞吐、显存、磁盘和完整运行 ETA，并经用户确认成本。

## 活动路线

1. Claude 从论文侧补齐带页码/图表锚点的 Nature 主张与协议；Codex 独立核对作者代码和执行实现。
2. 将核验事实收敛到 implementation ledger 与 claim-protocol matrix，discussion 只保留未决问题。
3. 协议候选闭合后再创建 replication 卡；smoke/pilot 只验证环境、仪器和成本，不形成性能结论。

## Parked Lanes

- 2013 独立重实现约 5M-update 长程扩展。
- 只给 2013 实现加入 target network 的单变量机制实验。
- 多训练 seed、其他 Atari 游戏和精确历史 ALE/Torch7 环境重建。
