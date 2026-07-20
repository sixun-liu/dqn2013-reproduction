# DQN Claim-Protocol Matrix

## 目标主张

Breakout 上的 2013 DQN 学到显著优于随机策略的行为；论文 Table 1 报告平均 168、best 225。

## 冻结协议

| 字段 | 本地选择 | 证据与限制 |
|---|---|---|
| Paper | arXiv `1312.5602`, SHA256 `8db04120cace173151c77e0faa6f3eaa4207009da66b9417597dc70bfee56d9c` | primary |
| Code | `src/dqn2013_breakout.py`, SHA256 `77aa269cd39888ebae4b0c256a120056141f96978284428801f2493e23ee62c0` | independent reimplementation |
| Scaffold reference | CleanRL `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`, MIT | Nature-style third party, 不作数值真值 |
| ALE | Gymnasium 0.29.1 / ALE-Py 0.8.1 / `BreakoutNoFrameskip-v4` | repeat probability 0，最小 4 动作 |
| Preprocessing | grayscale -> resize 110x84 -> crop rows 18:102 -> stack 4 | 对应论文文字；具体插值是现代实现选择 |
| Action repeat | 4，无 max-pooling | 论文明确 k=4；区别于 Nature wrapper |
| Terminal | 真实 game over，不把 life loss 当 terminal | 2013 算法口径 |
| Network | Conv 16@8/4, Conv 32@4/2, FC 256 | 论文 Section 4.1 |
| Target | online Q 的 stop-gradient bootstrap | 不使用 2015 target network |
| Optimizer | RMSProp lr 2.5e-4, alpha .95, eps .01 | 论文只明确 RMSProp；常数是历史实现惯例，非完整论文事实 |
| Replay | 250K agent transitions，约等于 1M emulator frames | 论文写 1M recent frames；frame/transition 语义有歧义 |
| Exploration | 1.0 -> 0.1 over 250K decisions；eval epsilon .05 | 换算为 1M emulator frames |
| Budget | 最多 2.5M decisions = 10M emulator frames | 保守按原始 emulator frames 解释；可能短于历史实现实际口径 |
| Evaluation | 每 250K decisions 跑 10K decisions，报告完整 episode mean | 论文 epsilon .05 / 10K steps；现代启动有 no-op/FIRE drift |
| Seeds | 0 | 单 seed 最多 `promising_unresolved` |

## 验收

- 工程成功：GPU update、replay、独立评估、JSONL 与 checkpoint 均有效。
- 部分复现：eval return 明显高于论文随机分 1.2，且训练/评估趋势持续上升。
- 数值接近：最终 eval 进入 168--225 仅作加强证据；现代协议漂移下不据此宣称 exact replication。

## 2026-07-17 Post-run Audit

本节不改写 EXP-0001 的冻结协议，只记录结果后的 provenance 复核。详见
`references/DQN2013_PROTOCOL_AUDIT.md`。

- Primary Figure 2 覆盖约 5M minibatch updates；EXP-0001 学习后只有约 2.45M updates，至少少一半。
- replay/epsilon 将论文的 1M `frames` 直接除以 repeat 得到 250K transitions/decisions；2015 作者
  代码与 NIPS 第三方复刻均按 1M agent time-steps 解释，当前换算缺少支持。
- 本地 optimizer/gamma/learning-start/no-op 是 2013 架构与 2015 常数的混合，不是恢复完整的
  2013 协议。
- 因此 EXP-0001 对“本地策略学习与退化”的观察有效，但与 Table 1 average 168 的直接数值比较
  provenance 不足；不得据此判定有效 2013 协议的数值复现为负。
- EXP-0002 固定 10 seed 复评确认最终 checkpoint seed mean 仅 2.21--2.46，排除了单一评估 seed
  偶然，但不解决训练协议归因。

# Nature 2015 Extended Data Table 3

## 目标主张

Nature 2015 Extended Data Table 3 报告 Breakout replay + target Q 路线的周期评估峰值平均分
`316.8`。本轮采用独立现代重实现，检验单 seed 学习曲线与该量级是否一致，不复现整张 2x2 表。

## 冻结前协议矩阵

| 字段 | 本地选择 | 证据与限制 |
|---|---|---|
| Paper | Nature 518 (2015), DOI `10.1038/nature14236`; SHA256 `cc811007a48aea14fcc135158ed96d01982930f415045a19f89474bfa1a74eb5` | primary；目标为 Extended Data Table 3 |
| Reference artifact | Breakout replay + target `316.8` | 表项是三种 learning rates 中的有利结果，原始逐点评估曲线未公开 |
| Code lineage | P2/A1 恢复协议；CleanRL MIT 骨架；本地 executor 为 `independent_reimplementation` | A1 受限源码只读，不复制进公开仓 |
| Task/environment | Gymnasium 0.29.1 / ALE-Py 0.8.1 / `BreakoutNoFrameskip-v4`; sticky=false | 与论文原 ALE 栈有版本漂移 |
| Preprocessing | no-op max30、两帧 max、repeat4、84x84 gray、stack4；train life-loss terminal，eval 只认 game over | wrapper 顺序和 FIRE reset 为现代实现选择 |
| Budget | 10M agent decisions = 40M emulator frames；约 2.4875M optimizer updates | 论文时间换算与 A1 计数器共同支持；不是 2.5M decisions |
| Training | replay 1M、warmup50K、batch32、update/4、target/10K、gamma .99；centered RMSProp 2.5e-4/.95/.95/.01；Huber delta1；reward clip | 逐项对齐 P2/A1；expanded config 在 replication freeze 固定 |
| Exploration | 1.0 -> .1 over 1M agent decisions | A1 `eps_endt` 由 decision 计数器消费 |
| Evaluation | 每250K train decisions；epsilon .05；135K eval decisions；无5分钟截断；完整 game mean；取轨迹 peak | 与 A1 默认125K eval存在差异时以论文 Table 3 为准 |
| Diagnostics | warmup 后固定约500 held-out states，记录 mean max-Q | 诊断，不替代主评估 |
| Repetitions | seed 0，固定 lr | 单 seed 且未做三 learning-rate 选择，最高 `promising_unresolved` |
| Acceptance | 工程信标全绿；报告 peak/final 全曲线并与316.8同图；不设伪精确容差 | 未完成10M预算为 `inconclusive` |
| Cost | pilot 100K--250K decisions；若10M ETA >20h则停止；不进入50M完整训练 | 用户批准的成本门 |

完整依据、漂移与禁止解释见 `references/NATURE2015_PROTOCOL_AUDIT.md`。
