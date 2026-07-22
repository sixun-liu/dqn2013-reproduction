# DQN 可解释性方法核验与本地适配

> Updated: 2026-07-22T03:20:00Z  
> Maintainer: codex  
> Scope: Nature 2015 DQN Breakout EXP-0004 的离线局部机制分析

## 研究问题

在固定的 Nature 2015 DQN 单 seed 基线上，训练阶段、价值估计、FC512 表征、视觉扰动敏感性与
完整游戏表现之间出现哪些可复算关系？哪些只是描述关联，哪些能由离线干预支持？

排除：不从 2013/2015 跨实现差异归因 Target Network；不把 saliency 或 t-SNE 单图当作解释；
不声称跨 seed 稳定性。

## Primary Sources

| ID | Source | 本地 PDF / SHA256 | 原文方法锚点 | 本地用途 |
|---|---|---|---|---|
| XAI-001 | van Hasselt et al., 2016, Double DQN, arXiv:1509.06461 | `/root/autodl-tmp/papers/interpretability/1509.06461.pdf`; `a37336d7...e496` | p.4--5, Eq. Double-DQN target, Fig.3 value estimate/discounted-return protocol | Q 分布与 Q-return calibration；不把 target network 等同于 Double DQN |
| XAI-002 | Greydanus et al., 2018, arXiv:1711.00138 | `/root/autodl-tmp/papers/interpretability/1711.00138.pdf`; `85449caa...5e63` | p.3--4, Eq.1 localized Gaussian blur, `k=5` saliency grid | 对 DQN Q 向量作扰动式敏感性适配 |
| XAI-003 | Zahavy et al., 2016, arXiv:1602.02658 | `/root/autodl-tmp/papers/interpretability/1602.02658.pdf`; `176d8921...1fb` | p.5--7, trajectory statistics、FC activation、PCA50+t-SNE | 表征探索图；不能单独裁决簇质量 |
| XAI-004 | Anand et al., 2019, AtariARI, arXiv:1906.08226 | `/root/autodl-tmp/papers/interpretability/1906.08226.pdf`; `5ba7be92...759` | p.4--5, RAM state variables 与 linear classifier mean F1 | 可选的表征可解码性 probe；按 episode 切分 |
| XAI-005 | Atrey et al., ICLR 2020, arXiv:1912.05743 | `/root/autodl-tmp/papers/interpretability/1912.05743.pdf`; `6bcbc1af...641` | p.1--4, saliency 是 hypothesis generator，需 semantic counterfactual | 强制正负对照和自然性边界 |
| XAI-006 | Wang et al., 2016, Dueling DQN, arXiv:1511.06581 | `/root/autodl-tmp/papers/interpretability/1511.06581.pdf`; `11239e34...a47` | p.2, output 对 input stack 的 Jacobian saliency | 梯度敏感性只作快速对照 |

六个 PDF 的完整 SHA256 已在 2026-07-22 本机复核。Claude 两份综合的稳定快照与 hash 见
`references/manifests/2026-07-22_dqn-interpretability-ingestion.tsv`。

## 对 Claude 规格的必要修正

1. van Hasselt Fig.3 比较训练期平均 value estimate 与最终 best policy 的平均实际折扣回报；
   本项目若对每个 checkpoint 计算自身策略 return，属于 stage-conditioned calibration adaptation，
   不是 Fig.3 数值复刻。
2. Target network 缓解 moving target；Double DQN 才明确拆开 max 的 action selection/evaluation。
3. EXP-0004 checkpoint 不保存 ALE/replay state。`heldout_states.npy` 只能做固定输入推理，不能从
   任意 observation 恢复环境 rollout。
4. Q-return 必须匹配 clipped reward、life-loss terminal 和 `gamma=0.99`；完整游戏 raw score
   只用于行为表现。
5. EXP-0001 仅有 final checkpoint；EXP-0003 有六个 2013-unitfix checkpoint，但网络、预处理、
   replay、优化器和预算均与 2015 不同。跨实现图只作混杂历史上下文。
6. t-SNE 对 seed/perplexity 敏感；必须给 PCA 对照与多 seed 稳定性，证据权限保持 illustrative。
7. AtariARI 需要新标注轨迹、依赖/许可审计和 episode-level split，不能把相邻帧随机拆分。

## 本地方法顺序

1. **固定状态价值/表征面板**：40 checkpoints x 500 states，提取 Q、action margin、FC512。
2. **阶段行为表**：复用 40 次 135K-decision 完整游戏评估及逐 episode returns。
3. **匹配语义的校准轨迹**：selected checkpoints 另采 clipped/life-loss/discounted-return trace。
4. **视觉干预**：固定状态的四帧消融、Gaussian blur saliency、随机区域对照。
5. **表征探索**：effective rank、linear CKA 为定量；PCA/t-SNE 为审查入口。
6. **AtariARI**：只有前五项无法回答表征语义时再接入，不作为当前完成门。

## 证据权限

- fixed-state paired statistics：支持该固定输入面板上的阶段差异。
- on-policy trajectory statistics：支持该 checkpoint/策略访问分布上的行为与校准描述。
- blur/frame intervention：支持局部输入扰动对 Q/action 的敏感性，不等于自然环境因果解释。
- stage association：只作描述关联；单条训练时间序列和单 seed 不支持机制因果或稳定性检验。
- 2013 comparison：只作 historical context。

唯一下一取证动作：先完成 EXP-0005 fixed-state panel 的 known-answer/parity gate，再决定是否采集
校准轨迹和 saliency。

