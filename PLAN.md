# PLAN

> Updated: 2026-07-17T05:26:23Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `reproduction`
- 北极星：理解 2013 DQN，并在现代可审计环境中复现 Breakout 的学习趋势。
- 当前主要矛盾：单位修正已解决首次失稳窗口，但长程稳定性和论文数值尺度仍未闭合。

## 阶段退出门

- [x] 固定 2013 paper、独立实现和 third-party 谱系。
- [x] 完成 10M emulator-frame 基线、固定 seed 复评和单位语义干预。
- [ ] 用户复核 `EXP-0001` 与 `EXP-0003` 证据图。
- [ ] 决定是否投入约 5M-update 长程单 seed 基线，或在当前部分复现处停止。

## 活动路线

1. 用户复核最终低分、Q/loss 爆冲和单位修正对照。
2. 在复核前不启动 GPU run，不测试 optimizer/gamma 或 target network。
3. 若继续，预注册从头运行的协议对齐长程基线；checkpoint 不作协议等价续跑。

## Parked Lanes

- NIPS optimizer/gamma bundle。
- 2015 target network 机制对照。
- 多训练 seed、其他 Atari 游戏和历史 ALE 环境重建。
