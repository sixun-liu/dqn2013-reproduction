# DEVLOG

> Updated: 2026-07-21T02:34:37Z
> Maintainer: codex
> Source of truth: decision synthesis linked to research IDs

只追加持久决策、协议变化、路线升降级、正式裁决和迁移；原始运行输出留在 registry/artifact。

## 2026-07-16

### 2026-07-16T18:27:00Z | protocol | EXP-0001

- Actor: codex
- Summary: 目标固定为 2013 arXiv DQN Breakout 独立重实现；CleanRL 仅作 2015 工程参照，运行使用 online target、两层卷积和 repeat 4。
- Evidence: `references/CLAIM_PROTOCOL_MATRIX.md`, source SHA256 `77aa269c...62c0`
- Next: 运行 10M emulator-frame 单 seed 基线并分层比较随机、学习趋势和论文分数。
- Git: historical pre-Git source hash; CleanRL `fe8d8a0`

### 2026-07-16T21:39:01Z | result | EXP-0001

- Actor: codex
- Summary: 评估 peak 10.90、final 2.21；策略学到过高于随机的行为但明显回退，论文 168 数值未复现。
- Evidence: EXP-0001, ART-0001--ART-0006
- Next: 先审计 frame/step、预算和评估语义，不原样重跑。
- Approval: pending-user-review

## 2026-07-17

### 2026-07-17T01:46:46Z | result | EXP-0002

- Actor: codex
- Summary: 最终 checkpoint 的 10 个固定评估 seed 均值为 2.21--2.46，末点退化不是单一 seed 偶然。
- Evidence: EXP-0002, ART-0007--ART-0011
- Next: 用受控协议干预解释首次退化窗口。

### 2026-07-17T01:53:46Z | protocol | EXP-0003

- Actor: codex
- Summary: parity smoke 逐 tensor 一致后，只把 replay capacity 和 epsilon decay 从 250K 修正到 1M agent steps。
- Evidence: EVT-0017; config SHA256 `53a20755...9907`; executor SHA256 `b8bea12b...16f6a`
- Next: 在 1.45M--1.5M 预注册窗口比较 Q/loss 和配对评估。
- Git: historical pre-Git source hash

### 2026-07-17T03:57:38Z | result | EXP-0003

- Actor: codex
- Summary: Q mean max 61.74→3.72，loss max 25.40→0.242，终点评估 2.51→10.04；联合单位修正通过支持门槛，裁决 `promising_unresolved`。
- Evidence: EVT-0025, ART-0012--ART-0018
- Next: 用户复核后决定是否从头运行约 5M-update 长程基线。
- Approval: pending-user-review

### 2026-07-17T05:04:25Z | migration | control-repo-bootstrap

- Actor: codex
- Summary: 将既有独立实现、控制材料和 registry 作为 post-run snapshot 首次 Git 化，不反向声称是 pre-run commit。
- Evidence: `MIGRATION.md`, `research/repositories.yaml`
- Next: 下一 formal/replication 从 clean commit 冻结。
- Approval: user
- Git: control `21cc937`; workflow `e656c16` (`v0.1.0`)

### 2026-07-17T05:31:57Z | workflow | control-doc-strategy

- Actor: codex
- Summary: 项目显式采用事件触发式控制文档规范；10 个完成项从 TODO 移出，DEVLOG 压缩为可追溯事件。
- Evidence: `researchctl docs --strict` 0 error/0 warning
- Next: 只在方向变化、持久决策或正式结案时更新相应人读视图。
- Approval: user
- Git: workflow `3a5bd50` (`v0.1.1`)

### 2026-07-17T05:43:36Z | workflow | multi-repo-provenance

- Actor: codex
- Summary: 项目采用 schema 2 repository manifest；未来 new/freeze 自动快照 control/runtime/workflow 与三个 third-party 仓。
- Evidence: `research/repositories.yaml`, workflow multi-repo regression tests
- Next: 下一 formal run 从 clean control commit 和固定 third-party commits 预注册、冻结。
- Approval: user
- Git: workflow `f3a120e` (`v0.2.0`)

### 2026-07-17T05:47:34Z | workflow | human-review-status-fix

- Actor: codex
- Summary: pending human review 继续保留 strict warning，但不再被误报为需要修复的审计损坏。
- Evidence: workflow 15/15 tests; `researchctl status`
- Next: 下一控制动作直接指向最早 pending review。
- Approval: user
- Git: workflow `ffc2d66` (`v0.2.1`)

## 2026-07-20

### 2026-07-20T16:18:38Z | decision | nature-2015-route

- Actor: user, codex
- Summary: 2013 独立重实现保留现有趋势与单位修正证据，不再投入长程扩展；Nature 2015
  Breakout 升为当前主线，并先退回 understanding 完成论文、代码、ALE、评估和成本对账。
- Evidence: EXP-0001--EXP-0003；DeepMind DQN 3.0 `9d9b1d1`；实现谱系审计。
- Next: 选择现代 executor 并冻结 Nature 2015 claim-protocol 后，只运行最小 smoke/pilot。
- Approval: user
- Git: control@`515a40c` before transition; workflow@`ffc2d66` (`v0.2.1`)

### 2026-07-20T17:27:07Z | protocol | nature-2015-table3

- Actor: codex
- Summary: 首目标冻结为 Nature Extended Data Table 3 Breakout replay + target `316.8`；预算按
  10M agent decisions = 40M emulator frames，评估每250K decisions运行135K decisions并取轨迹峰值。
- Evidence: `references/NATURE2015_PROTOCOL_AUDIT.md`, `references/CLAIM_PROTOCOL_MATRIX.md`,
  Nature PDF SHA256 `cc811007...eb5`, DeepMind DQN 3.0 `9d9b1d1`。
- Next: 实现带 MIT attribution 的独立 PyTorch executor 和聚焦测试，再用100K--250K pilot实测成本。
- Approval: user approved autonomous option 2 and cost gates

## 2026-07-21

### 2026-07-21T02:34:37Z | result | EXP-0004

- Actor: codex
- Summary: Nature 2015 Breakout 正式 run 自然完成；40 次评估 peak/final mean 均为 350.18，超过论文 Table 3 的 316.8 参考值 10.54%，裁决 `promising_unresolved`。
- Evidence: EVT-0028--EVT-0032, ART-0019--ART-0027
- Next: DQN 计算路线在单 seed 部分数值复现处停止；第二 seed 问题 parked，转入 DreamerV3 与双论文交付整合。
- Approval: human visual review pending
- Git: frozen runtime@`d1234d9`; result branch@`f2c15be`
