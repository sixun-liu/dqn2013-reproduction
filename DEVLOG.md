# DEVLOG

> Updated: 2026-07-17T05:31:57Z
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
