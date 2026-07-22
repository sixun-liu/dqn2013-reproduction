# EXP-0005 DQN 离线定量与视觉干预协议

> Status: protocol draft before experiment freeze  
> Updated: 2026-07-22T03:20:00Z  
> Maintainer: codex

## 1. 问题与证据权限

问题：固定 EXP-0004 seed0 Nature DQN 后，价值分布、action margin、FC512 表征、视觉扰动敏感性
和评估行为如何随训练阶段共同变化？

本周期是 `probe`。它可以支持固定任务、固定 seed、固定状态/轨迹上的描述与输入干预效应；不能
支持 Target Network 因果、跨 seed 稳定性或 2013/2015 单变量归因。

## 2. 冻结来源

- Run：`/root/autodl-tmp/runs/EXP-0004__breakout__s000__10m-dec__20260720T174600Z`
- Runtime source：`src/dqn2015_nature_breakout.py`，训练 commit `d1234d94c106c87d61c6baa3a090a38802a7df76`
- Config：`configs/nature2015_table3_s0_formal_10m.json`
- Checkpoints：250K 到 10M，每 250K 一个 evaluation checkpoint；10M 使用 `complete` checkpoint，
  避免同 step 双文件。
- Fixed states：run 下 `heldout_states.npy`，500 x 4 x 84 x 84 uint8。
- Behavior：run 下 `metrics.jsonl` 的 40 条 `evaluation` 和逐 episode returns。
- Analysis output：`/root/autodl-tmp/artifacts/dqn2013/EXP-0005/`
- Review output：`/root/autodl-tmp/artifacts/dqn2013/review/EXP-0005-dqn-offline-atlas/`

所有输入在执行前写绝对路径、bytes、SHA256 和 checkpoint decision 到 `source_manifest.json`。

## 3. 数据层

### A. Fixed-State Panel

20,000 行：40 checkpoints x 500 state IDs。每行至少包含：

```text
checkpoint_decisions, checkpoint_sha256, state_id, observation_sha256,
q_noop, q_fire, q_right, q_left, max_q, second_q, action_margin,
greedy_action, feature_l2
```

FC512 activation 单独保存为 float32 NPZ：`features[40,500,512]`。Q 保存为
`q_values[40,500,4]`，同时导出轻量 CSV summary。

Known-answer/parity gates：

1. 40 个 checkpoint decision 唯一且严格递增；最终 checkpoint hash 匹配 EXP-0004 registry。
2. states shape/dtype/hash 固定，Q/features 全有限。
3. 每个 checkpoint 重算 `mean(max_q)` 与原 metrics 同 step heldout-Q；绝对误差 `<=1e-5`。
4. 手工前向 `network[:-1] -> final linear` 与完整 forward 的 Q 最大误差 `<=1e-6`。

### B. Behavior Panel

直接解析 40 条冻结 evaluation：stage、完整 game 数、mean/median/min/max、逐 episode return 和
interrupted。该表使用 full-game/raw-reward 语义，不与训练 Q 直接相减。

### C. Calibration Trajectories

只在 A/B 通过后，从 0.25M、1M、2.5M、5M、7.5M、10M 六个 checkpoint 采轨迹。单独使用与
训练一致的 `EpisodicLifeEnv + ClipRewardEnv + repeat4 + FrameStack4`，greedy policy，记录每个
pseudo-episode 的 clipped reward，并计算：

```text
G_t = sum_{k>=0} gamma^k r_{t+k}, gamma=0.99
calibration_gap_t = max_a Q(s_t,a) - G_t
```

完整标量 trace 全保存；图像只按结果盲的固定 RNG 每阶段 reservoir 256 states。`G_t` 是当前
greedy policy 的 sample return，不是 `Q*`，gap 不能全部解释为 maximization bias。

### D. Visual Interventions

Selected stages 与 C 相同。四帧消融覆盖 fixed 500 states；空间 saliency 使用预先固定的 128
state IDs。空间扰动按 Greydanus localized Gaussian blur：mask variance 25、blur sigma 3、grid
stride 5，并同时作用于四个 stack channels。

保存：Q-vector L2 change、原 greedy action 的 delta-Q、action switch、saliency map、map entropy
和 top-decile concentration。梯度 saliency 只作方向对照。随机 grid-cell 与全局 blur 是负/正
敏感性控制；没有控制通过时不解释空间语义。

## 4. 表征统计

对固定状态 FC512：

- covariance eigenvalue effective rank 与 participation ratio；
- 每阶段到 10M 表征的 centered linear CKA；
- PCA explained variance；
- selected stages 的 PCA50+t-SNE，冻结 perplexity 与三个 random seeds，只作图审。

不同 checkpoint 的 ReLU unit 没有显式对齐，不能逐 neuron 解释；CKA/谱指标只描述整体几何。

## 5. 统计计划

- 状态是 fixed panel 的 paired unit；阶段差异使用 state-ID paired bootstrap，2,000 resamples，
  seed `20260722`，报告 median difference、effect size 和 percentile 95% CI。
- calibration/saliency 轨迹按 pseudo-episode cluster bootstrap；相邻帧不作独立样本。
- 40-stage 关联报告 Spearman rho 与 checkpoint-block bootstrap（block length 5），明确为描述关联。
- selected-stage 对 10M 的五个预注册比较使用 Holm correction；不作看图后追加显著性检验。
- 单训练 seed 不报告 training-seed standard error，不用 episode CI 支持跨 seed 主张。
- 报告 median、IQR、p95/p99、最差状态/episode 和恢复阶段，不用单均值覆盖尾部。

## 6. 预注册替代解释

- 早期 heldout states 来自 50K replay，可能更接近早期策略分布；最终策略状态需由 C 补充。
- Q/representation 与 score 同步变化可能由共同训练进度驱动，不构成相互因果。
- saliency 对 blur 这一 OOD 干预敏感，不等同于对象语义或自然环境反事实。
- effective rank/CKA 变化可能来自尺度、ReLU sparsity 或状态覆盖，不自动等于表征更好。
- evaluation 的 epsilon=.05 full-game 行为与 calibration 的 greedy life-loss 轨迹口径不同，只能分栏。

## 7. 停止与完成门

立即停止并保留现场：输入 hash/step 不符、parity 超阈、NaN/Inf、GPU 重复进程、输出预计超过
2 GiB，或 calibration 环境 wrapper 与训练语义不等价。

完成信号：A/B/C/D 均有 manifest 和机器表；价值/行为、表征、视觉干预各至少一张同坐标审查图；
统计 summary 明确 CI、采样单位和限制；结案只留下一个是否值得运行 Target Network 因果消融的
判别问题。

