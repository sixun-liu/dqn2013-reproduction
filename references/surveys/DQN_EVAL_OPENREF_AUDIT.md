# DQN 评估 wrapper 开源建议核验

> Updated: 2026-07-22T04:48:00Z
> Maintainer: codex
> Scope: Claude openref 交接、EXP-0006/0007 calibration 失败与本机执行栈

## 来源与版本

| 材料 | 类别 | SHA256 / 版本 | 核验结果 |
|---|---|---|---|
| Claude openref 交接 | `internal_synthesis` | `6321c279...8df` | 已稳定快照；建议需按下述边界解释 |
| `/root/autodl-tmp/papers/dqn_openref/cleanrl_dqn_atari.py` | `third_party` | `84ec3637...4f2` | 与 pinned CleanRL commit `fe8d8a0` 的文件字节一致 |
| `/root/autodl-tmp/papers/dqn_openref/sb3_atari_wrappers.py` | `third_party` | `4e1a42ae...14c` | 与本机 SB3 2.3.2 文件 hash 不同 |
| 本机 SB3 2.3.2 `atari_wrappers.py` | runtime dependency | `ba1c8cd7...29d` | 相关 reset 行为与下载版本一致；diff 仅 typing/文档 |

本项目 `src/dqn2015_nature_breakout.py::make_atari_env` 已直接导入 SB3 2.3.2 的
`NoopResetEnv`、`FireResetEnv`、`EpisodicLifeEnv`、`MaxAndSkipEnv` 和 `ClipRewardEnv`，没有
自行重写标准 wrapper。

## 已确认的建议

1. 标准 `NoopResetEnv.reset` 在 reset no-op 遇 terminal 时重新 reset；`FireResetEnv.reset` 对
   FIRE 两步分别检查 terminal；`EpisodicLifeEnv.reset` 对 life-loss 后的推进 no-op 再检查真实
   game over。本地 2.3.2 和下载版本在这些行为上相同。
2. 固定 decision budget 与固定完整 game 数不应同时作为必须达到的完成门。EXP-0006/0007 已实测
   纯 greedy 策略的完整游戏长度跨 checkpoint 相差至少一个数量级。
3. 训练校准语义必须保留 clipped reward、life-loss terminal 和 `gamma=0.99`；完整游戏 raw
   return 只能分栏报告。

## 不可直接套用之处

1. `post_fire_noop_burnin` 是为了打破无 sticky、纯 greedy Breakout 的完全确定性而加在
   **FireReset 之后** 的实验性初始状态干预，不属于标准 `NoopResetEnv`。遇 terminal 后静默 reset
   重采会条件化到“幸存初始状态”，并破坏四阶段共享相位的配对设计。
2. `EpisodicLifeEnv` 让训练 pseudo-episode 在 life loss 处终止，但不会让用于 cluster bootstrap
   的真实完整游戏变短。不能用 life 数冒充独立完整游戏数。
3. EXP-0006/0007 的部分 trace 未满足四阶段共同完成门，禁止用来裁决 Q-G 或 margin 假说。

## 路线裁决

当前 calibration 问题保持 `unknown`，不再继续技术重试。后续固定状态视觉干预可以复用作者或
第三方开源实现，但必须固定 commit/license、核对 blur 定义，并继续将扰动敏感性与自然语义因果
分开。若将来重开 calibration，应另建 cycle，在运行前选择以下一个口径：

- 固定 decisions，接受每阶段不同的完整游戏数并只作相应描述；或
- 固定完整 games，不设置紧 decision cap，并单独预估每个 checkpoint 成本。
