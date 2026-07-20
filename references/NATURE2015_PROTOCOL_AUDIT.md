# Nature 2015 DQN Breakout 协议审计

> 核验日期：2026-07-21  
> 目标：以可承受成本复现 Extended Data Table 3 的 replay + target 路线，不宣称复现整篇论文。

## 1. 主来源与代码谱系

| ID | 身份 | 稳定版本 | 用途 |
|---|---|---|---|
| P2 | Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518, 529--533 (2015), DOI `10.1038/nature14236` | `/root/autodl-tmp/papers/nature14236.pdf`; SHA256 `cc811007a48aea14fcc135158ed96d01982930f415045a19f89474bfa1a74eb5` | 论文主张、公开协议与参考数值 |
| A1 | DeepMind DQN 3.0 | commit `9d9b1d13a2b491d6ebd4d046740c511c662bbe0f`; limited academic review license | 作者代码语义的只读 oracle；不得复制源码进公开仓 |
| T2 | CleanRL `dqn_atari.py` | commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`; MIT | 现代 PyTorch 工程骨架 |
| L2 | 本项目 Nature executor | 待实现并由 replication freeze 固定 | `independent_reimplementation` |

论文优先于代码；作者代码用于恢复论文省略或含糊的计数语义；CleanRL 不是论文协议真值。

## 2. 目标主张

Nature 论文 Extended Data Table 3 在 Breakout 上报告：使用 replay 与 target Q network 的 DQN，
训练 10M 后所得周期评估最高平均 episode score 为 **316.8**。同表其他结果为 replay/no-target
`240.7`、no-replay/target `10.2`、二者均无 `3.2`。

本项目只执行 replay + target 的单 seed、现代 ALE 路线。目标是检验曲线是否稳定学习并接近论文
量级；不本地重做 2x2 消融，也不把历史 `EXP-0001` 当作 Table 3 的 no-target 对照。

## 3. 预算语义裁决

- 论文称完整训练为 50M `frames`，并称其约等于 38 天游戏经验。60 Hz 下 38 天约为 197M
  emulator frames，与 action repeat 4 后的 **50M agent decisions = 200M emulator frames** 一致。
- A1 `run_gpu` 设置 `steps=50000000`、`actrep=4`；训练循环每次环境决策令 `step` 加一，并打印
  `frames=step*actrep`。A1 的 epsilon 终点 `eps_endt=1000000` 也由同一 decision 计数器消费，且
  从 replay warmup 结束后开始退火。
- 因而 Table 3 的 10M 预算在本项目冻结为 **10M agent decisions = 40M emulator frames**，不是
  2.5M decisions。训练 epsilon 在 50K warmup 后的 1M agent decisions 从 1.0 线性退火到 0.1。
- 所有配置和指标都分别命名 `agent_decisions`、`emulator_frames` 与 `optimizer_updates`；孤立的
  `frames` 不作为预算变量。

## 4. 冻结训练协议

| 字段 | 本地协议 | 来源与说明 |
|---|---|---|
| Task/ALE | Gymnasium 0.29.1、ALE-Py 0.8.1、`BreakoutNoFrameskip-v4`、sticky actions false | 现代兼容选择；与原 ALE 版本有不可消除漂移 |
| Input | 相邻两帧 max、灰度、84x84、stack 4、action repeat 4 | P2 Methods / Extended Data Table 1 |
| Train terminal | 丢命作为 replay terminal；真实 game over 后 reset | A1/CleanRL wrapper 语义；评估不使用 life-loss terminal |
| Network | Conv 32@8/4, 64@4/2, 64@3/1, FC512, linear action head | P2 Methods；Breakout 4 动作时参数量 `1,686,180` |
| Replay | capacity 1,000,000 decisions；warmup 50,000；batch 32 | P2 Extended Data Table 1 |
| Update | 每 4 decisions 一次；gamma .99；target hard sync 每 10,000 decisions | P2 Extended Data Table 1 |
| Optimizer | centered RMSProp，lr 2.5e-4；gradient 与 squared-gradient momentum 均 .95；denominator addition/floor .01 | P2 Methods 与 A1 数值语义 |
| Loss | TD error 在 ±1 外使用常数梯度，即 smooth-L1/Huber delta 1 | P2 Methods；不得替换成 CleanRL 默认 MSE |
| Rewards | clip 到 [-1, 1] | P2 Methods |
| Exploration | warmup期间epsilon 1.0；随后在1M decisions内降至.1 | P2 表格 + A1 decision 计数器 |
| Random start | reset 后至多 30 个 no-op；需要 FIRE 的游戏执行 FIRE reset | P2 / A1；现代 wrapper 行为记录为 drift |

## 5. 冻结评估协议

- 每 250K training decisions 保存可评估 checkpoint 并启动独立 evaluator。
- epsilon 固定 .05；评估 **135K agent decisions**；不把 life loss 当 terminal，也不使用训练 return。
- Extended Data Table 3 明确评估不施加 5 分钟 episode 截断；统计该窗口内完成的真实 games 的
  arithmetic mean。窗口末未完成 episode 不进入均值。
- 论文表项取训练轨迹上周期评估的**最高平均分**，不是最后 checkpoint 分数。
- 同时在 warmup 后固定约 500 个 held-out replay states，记录各状态 `max_a Q(s,a)` 的均值，作为
  数值与价值估计诊断；它不是主验收分数。

## 6. 可比性与验收边界

论文 Table 3 为三个 learning rates 中的有利选择，原 ALE 版本、随机性和 seed 汇总细节不足以在
现代栈严格复原。因此 `316.8` 是参考线，不设伪精确的等值容差：

1. 工程门：环境、replay、GPU update、target sync、独立评估、held-out Q、JSONL 与 checkpoint
   全部有效且数值有限。
2. 成本门：100K--250K decisions pilot 实测后，10M decisions ETA 不超过 20 小时；否则以成本不适配
   关闭并转 DreamerV3。
3. 科学观察：报告完整周期曲线、peak 与 final，不因单点接近 316.8 选择性宣称成功。
4. 单 seed、现代 ALE 的最高裁决为 `promising_unresolved`；未完成预算为 `inconclusive`；协议或
   评估失效为 `invalid_provenance`。

完整 Nature Table 2 的 Breakout `401.2 +/- 26.9` 使用 50M decisions 和 30 个最多 5 分钟 episode，
仅作背景，不属于本轮计算目标。

## 7. 已知漂移与禁止解释

- Python/PyTorch/Gymnasium/ALE-Py 代替 Lua/Torch7/Xitari/ALE 原栈，属于执行环境漂移。
- 本地只跑固定 learning rate、seed 0；不能复现论文的 learning-rate 选择过程或不透明重复统计。
- `replay,no-target=240.7` 是 Nature 内部消融，不等于完整 2013 DQN；历史 `EXP-0001` 只作教学
  上下文，不支持“target network 单独修复本地退化”的因果结论。
- smoke 与 pilot 只裁决工程和成本，不进入论文结果 scoreboard。
