# DEVLOG

> Updated: 2026-07-22T10:44:39Z
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

## 2026-07-22

### 2026-07-22T03:20:00Z | decision | dqn-offline-mechanism-exploration

- Actor: user, codex
- Summary: 项目从 Nature 单 seed 结果复现进入 exploration；冻结 EXP-0004 为唯一 baseline，优先复用 40 个 checkpoint、500 个固定状态和评估轨迹建立价值、表征、视觉敏感性与行为的离线局部证据。
- Evidence: `references/DQN_OFFLINE_ANALYSIS_PROTOCOL.md`；六篇 primary PDF hash；Claude 两份 `internal_synthesis` ingestion manifest。
- Next: 创建 EXP-0005 probe，并先通过 fixed-state Q/FC512 panel 的 known-answer 与 held-out-Q parity。
- Approval: user
- Git: branch `exp/EXP-0005-dqn-offline-atlas`; baseline runtime `d1234d9`

### 2026-07-22T03:55:59Z | result | EXP-0005

- Actor: codex
- Summary: 40 x 500 固定状态 Q/FC512 面板通过零误差 parity；Q 绝对尺度不是行为质量的充分
  代理，`9.25M` 的高 Q、低 margin、低回报异常把下一轮从平均六阶段收敛到四个局部阶段。
- Evidence: EVT-0034--EVT-0038, ART-0028；原始阶段 rho(max-Q/margin/CKA)=
  `0.846/0.850/0.861`，一阶差分为 `0.056/0.317/0.085`。
- Next: 用 `9.0M/9.25M/9.5M/10M` 的匹配训练语义 calibration trajectory 区分 Q 尺度尖峰、
  状态访问与评估方差；信号通过后才做视觉干预。
- Approval: user approved autonomous work; human visual review pending
- Git: extraction freeze@`c88f828`; post-run atlas@`73f8a43`

### 2026-07-22T04:47:24Z | route_closed | EXP-0006--EXP-0007

- Actor: codex
- Summary: 匹配训练语义的纯 greedy calibration 两级成本门均未形成四阶段共同面板；9.5M 在
  300K decisions 内仅完成17/23局，按预注册停止重试，所有部分 Q-G trace 排除。
- Evidence: EVT-0039--EVT-0048；失败现场 `EXP-0006-failed-*`、`EXP-0007-failed-cap300k`。
- Next: Q-G 保持 unknown；转向固定状态四帧消融与作者公式 blur perturbation。
- Approval: user approved autonomous work; no human review required for failed completion signal
- Git: EXP-0007 freeze@`c3058b2`

### 2026-07-22T04:50:00Z | decision | author-interpretability-code

- Actor: user, codex
- Summary: 优先参考作者开源实现。Greydanus `182492d` 为当前视觉干预公式 oracle；AtariARI
  `a06f52c` parked；Atrey `500aefc` 因无 license 文件与 Toybox 依赖仅只读参考。
- Evidence: `references/surveys/DQN_EVAL_OPENREF_AUDIT.md`、`references/IMPLEMENTATION_LEDGER.md`。
- Next: 现代化 Greydanus `r=5,d=5,sigma=3` 批处理接口并先做单状态 parity。
- Approval: user

### 2026-07-22T05:28:49Z | result | EXP-0008

- Actor: codex
- Summary: Greydanus 作者方法的 `4 checkpoints x 128 states x 289 masks` 干预通过 parity；
  9.25M 的 local/global/frame 敏感度未高于三个对照，广泛视觉脆弱假说判为 `negative`。
- Evidence: EVT-0051--EVT-0058, ART-0029, ART-0031；9.25M local switch 配对差区间均跨 0，
  但 mean margin 最低、top-decile spatial concentration 最高。
- Next: 先合并 EXP-0004/0005/0008 为同一四阶段证据图，再决定 margin 匹配或对象/RAM probe。
- Approval: human visual review pending
- Git: extraction freeze@`e33f000`; review@`77eff19`

### 2026-07-22T05:36:29Z | synthesis | C-0001

- Actor: codex
- Summary: 将 EXP-0004 行为、EXP-0005 Q/FC512 与 EXP-0008 视觉干预按四个共同 checkpoint
  合并；9.25M 的局部异常固定为高 Q、低 margin、低 CKA/rank、低扰动幅度和高空间集中度共现。
- Evidence: `ART-0032`, `C-0001`；输入/output hash 与生成脚本 commit `460a0d2` 已入 manifest。
- Next: 用户复核后只在 margin 匹配、对象/RAM probe、Target Network 消融中选择一条新路线。
- Approval: human visual review pending

### 2026-07-22T10:44:39Z | decision | public-release-v0.1.0

- Actor: codex
- Summary: Nature 2015 DQN Breakout 单任务、单 seed 部分数值复现以 MIT 代码和中文报告公开；
  默认验证环境固定为 CPU-only，CUDA 仅由完整训练的显式 `setup-gpu` 安装。
- Evidence: GitHub Release `v0.1.0`; checkpoint SHA256 `73e3e71f...d672ef0`; 公开下载文件 CPU
  复评 135K decisions 得到 63 games、mean `345.22`、median `377.0`。
- Next: 等待用户/导师审阅报告；不因发布自动追加 DQN 训练或机制实验。
- Approval: user approved MIT license and checkpoint publication
- Git: main@`7ebbf1f`; tag `v0.1.0`; release branch@`debfa3e`
