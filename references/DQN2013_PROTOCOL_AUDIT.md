# DQN 2013 协议与故障审计

日期：2026-07-17。问题：EXP-0001 的低分和末段回退，分别有多少来自实现错误、协议漂移、评估
方差或算法动态？本审计是 control task，不创建论文结果 claim。

## 来源与权限

| ID | 来源 | 类别 | 版本与限制 |
|---|---|---|---|
| P1 | Mnih et al. 2013, arXiv `1312.5602` | primary | PDF SHA256 `8db04120...56d9c` |
| A1 | `google-deepmind/dqn` | author | commit `9d9b1d1`，仅对应 Nature 2015；limited academic license，不能冒充 2013 代码 |
| T1 | `spragunr/deep_q_rl` `run_nips.py` | third_party | commit `3420249`，BSD；明确以 NIPS 2013 参数为目标，但作者也承认原论文存在 blank areas |
| T2 | CleanRL `dqn_atari.py` 与文档 | third_party | commit `fe8d8a0`，MIT；对应 2015 Nature-style DQN |
| L1 | 本地 `src/dqn2013_breakout.py` | independent_reimplementation | SHA256 `77aa269c...62c0`；EXP-0001 冻结源码 |

稳定代码与 license hash 见 `IMPLEMENTATION_LEDGER.md`。

## Primary 可确认事实

- P1 Algorithm 1 每个 agent time-step 存一条 transition、采一个 minibatch 并更新一次；online Q
  生成 bootstrap target，不含独立 target network（论文正文第 4 节、Algorithm 1）。
- 预处理为 gray -> 110x84 -> crop 84x84 -> stack 4；网络为 16@8/4、32@4/2、FC256。
- reward 在训练时 clip 到 -1/0/1；RMSProp、batch 32；epsilon 在前 1M `frames` 从 1 降到 .1；
  replay 为最近 1M `frames`；总训练 10M `frames`；action repeat k=4。
- Figure 2 用 epsilon=.05 跑 10K `steps`；每个 epoch 是 50K minibatch weight updates，图横轴到
  100 epochs，即图中覆盖约 5M updates。
- Table 1 的 Breakout DQN average 为 168，best episode 为 225；average 只说明固定步数评估，
  正文没有给出足以在现代 ALE 中逐项重建的完整 wrapper/seed/optimizer 常数。

## 协议对账

| 字段 | P1 / NIPS third-party T1 | EXP-0001 本地 | 审计判断 |
|---|---|---|---|
| Network / online target | 2013 两层网络；无 target network | 匹配 | 对齐 |
| Preprocess | crop 84x84、stack4；T1 还实现了两帧 max | crop 84x84、stack4、无 max | primary 未写 max；本地选择可辩护 |
| Reward / terminal | clip reward；真实 game over | 匹配 | 对齐 |
| Action repeat | 4 | 4 | 对齐 |
| Replay capacity | P1 写 1M frames；T1 与 A1 都保存 1M agent time-steps | 250K transitions | 高风险单位漂移，容量缩小 4 倍 |
| Epsilon decay | P1 写前 1M frames；T1/A1 均按 1M agent steps | 250K decisions | 高风险单位漂移，探索退火快 4 倍 |
| Budget / updates | Figure 2 约 5M updates；T1 为 100x50K=5M agent steps | 2.5M decisions，学习后约 2.45M updates | 至少比 Figure 2 少约一半更新；不能直接对齐图或 Table 1 |
| Optimizer | P1 只写 RMSProp；T1: lr .0002, gamma .95, rho .99, eps 1e-6 | lr .00025, gamma .99, alpha .95, eps .01 | 本地混入 2015 常数；2013 exact 值仍 unknown |
| Replay start | P1 unknown；T1 为 100；A1 为 50K | 50K | 来自 2015 谱系，非 2013 primary 事实 |
| Random starts | P1 unknown；T1 NIPS 为 0；A1 Nature 为 30 | 30 + FIRE reset | 现代/Nature wrapper 漂移 |
| Evaluation | P1 epsilon=.05 / 10K steps | 每点 10K decisions，但每点换 seed | 步数大体匹配；跨 checkpoint 未做 paired seed |
| Stability metric | P1 固定 random-policy held-out states 的 max-Q | 当前 minibatch `q_mean` | 不能与 Figure 2 的 Q 曲线比较 |
| Checkpoint | P1 未规定 | 只保留 latest，不含 replay | 无法同 seed 复评 9M checkpoint，也不能协议等价续训 |

## EXP-0001 故障时间线

- 1M--5M raw emulator frames：评估 1.30 -> 2.37 -> 6.75 -> 8.47 -> 7.08；batch Q 中位
  约 0.97 -> 1.89，未见大爆冲。
- 5.8M--6.0M：首次明显异常，logged batch `q_mean` 最大 61.74、loss 最大 25.40；6M 评估降到
  2.51。由于日志只是每 1K decisions 的一个 batch，这些是下界，不是完整峰值。
- 6M--8M：继续出现 Q 41.75/38.05 等尖峰，评估保持约 2.5--3.2。
- 9M：单点评估恢复到 10.90；10M 又降到 2.21。
- EXP-0002 对最终 checkpoint 做 10 个固定 seed 复评：seed mean 2.21--2.46，排除单个评估
  seed 偶然，确认最终策略普遍退化。

## 裁决

1. **工程证据有效**：当前实现确实学习过，并在本地协议下发生真实的末点策略退化。
2. **论文数值比较 provenance 不足**：EXP-0001 的 replay、epsilon 和更新预算没有闭合到 P1/T1
   口径，因此 2.21 vs 168 只能说明本地 run 未对齐，不能判定 2013 DQN 在有效协议下复现失败。
3. **不能把责任直接归给 online target**：Q/loss 爆冲与性能回退时序一致，但同时存在 buffer、
   exploration、optimizer、gamma、wrapper 和训练 seed 差异。
4. **最高价值下一测试**：保持旧源码其余选择不变，只把 replay capacity 和 epsilon decay 从
   250K 改为 1M agent steps，在旧 run 首次异常对应的 1.5M decisions 处比较 Q 尖峰和同 seed
   evaluation；若仍不稳定，再测试 NIPS optimizer/gamma bundle。

## 不应立即做的事

- 不原样重跑 EXP-0001；它不能减少主要协议不确定性。
- 不直接加入 target network 后声称修复了 2013 DQN；那是 2015 机制对照。
- 不一次同时改变 buffer、epsilon、gamma、optimizer、no-op、max-pool 和 target network；即使分数
  上升也无法归因。
