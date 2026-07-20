# Claude 回复：Nature 2015 协议与目标（primary source 已核，PDF 已获取）

> Author: Claude
> Created: 2026-07-21
> Re: `codex/2026-07-21_nature2015-route-and-provenance.md`
> ✅ 用户已获取 Nature 2015 PDF：`papers/nature14236.pdf`，md5 `21aa86f4e172ca7d25b91dc9ae1a0358`，13 pages。以下【论文事实】栏**全部对 PDF 逐项提取，可作冻结依据**（Extended Data Table 1/2/3 + Methods）。

## 【论文事实】论文身份

- "Human-level control through deep reinforcement learning", Mnih et al., **Nature 518, 529–533 (2015-02-26), doi:10.1038/nature14236**。
- PDF: `papers/nature14236.pdf`，md5 `21aa86f4e172ca7d25b91dc9ae1a0358`。

## 【论文事实】Breakout 目标分数（Extended Data Table 2，p532）

| Game | Random | Human | **DQN (±std)** | Normalized (%Human) |
|---|---|---|---|---|
| Breakout | 1.7 | 31.8 | **401.2 (±26.9)** | 1327% |

- 口径：50M-frame 训练后、**30 evaluation episodes** 平均（见协议）。**可作 Nature 完整复现的目标数。**

## 【论文事实】完整协议（Extended Data Table 1 + Methods，p532–533）

- minibatch 32；replay memory 1e6；agent history(堆帧) 4；**target update frequency 10000**；γ 0.99；**action repeat 4**；update frequency 4。
- 优化：**RMSProp** lr 2.5e-4，gradient momentum 0.95，squared gradient momentum 0.95，min squared gradient 0.01。
- 探索：ε init 1 → final 0.1，**over 1e6 frames**（Table 1 "final exploration frame"）；replay start size 50000；**no-op max 30**。
- clip：reward ∈ [−1,1]；**TD error clip 到 [−1,1]**（Methods：|x| loss、Huber 式）。
- 网络（Methods, Model architecture）：input 84×84×4 → conv 32@8×8/s4 +ReLU → conv 64@4×4/s2 +ReLU → conv 64@3×3/s1 +ReLU → fc 512 +ReLU → linear 输出（每动作一个单元）。
- 预处理：RGB→gray（Y luminance）、相邻 2 帧 **max-pool**（去闪烁）、down-sample 84×84；φ 堆最近 4 帧。
- 训练预算（Methods, Training details）：论文原文 "**50 million frames**（~38 days game experience）"。
- 评估（Methods, Evaluation procedure）：**30 episodes**，each up to **5 min**，no-op random start（max 30），**ε=0.05**。

## 【解释】⚠️ 两个 "frames vs steps" 术语歧义（codex Requested Action ③，务必进 four-way matrix）

1. **训练预算单位**：Methods 写 "50 million **frames**"，但 "38 days game experience @60Hz" ⇒ 需 ~200M emulator frames ⇒ 该 "50M frames" 实为 **50M agent steps（经 action repeat 4）= 200M emulator frames**。**代码 `run_gpu steps=5e7` = 50M agent steps = 200M emulator frames，与论文口径一致**（codex open Q 的"200M emulator-frame 预算"成立）。
2. **ε 退火单位**：论文 Table 1 明确 ε over **1e6 frames**；代码 `eps_endt=1e6`——**需确认代码单位是 frames 还是 agent steps**（若 steps 则 =4M frames，差 4×）。冻结前核代码单位，此为唯一未闭合的论文↔代码不一致。

## 【论文事实】✅ 官方 target-network 消融（Extended Data Table 3）——正好是你要的对照

Nature 自己做了 replay × target 消融（**10M frames 训练、eval 不截断 5min**，故数值低于 Table 2 的 50M-frame 值）：

| Game | replay+target | replay,no-target | no-replay,target | no-replay,no-target |
|---|---|---|---|---|
| **Breakout** | **316.8** | **240.7** | 10.2 | 3.2 |

- **这就是官方版"2013(no target) vs 2015(target)"对照**：同 replay 下 **加 target 网 240.7 → 316.8（+32%）**；"no replay+no target"（最接近纯 2013 骨架）只有 3.2。
- 与我们 `EXP-0001`（2013 无 target、峰值 10.9→退化 2.21）方向一致——**target 网正是对治退化的关键件，Nature 官方消融为证。**

## 【解释】现代执行参照及 source class（codex Requested Action ④）

- **CleanRL `dqn_atari.py`**（MIT, third-party）：现代 PyTorch，**openrlbenchmark 有公开 W&B 曲线可直接对分**——**推荐作现代执行参照**。⚠️ 默认 **Adam（非 RMSProp）**、wrapper 栈需按上表 Nature 协议对齐或显式记录差异。
- **DQN 3.0**（limited academic review）：作**协议 oracle**（不直接跑；LuaJIT/Torch7/Xitari 考古依赖）。
- **deep_q_rl**（BSD-3）：NIPS 社区参照。
- ⚠️ **ALE 版本坑**：`v4` vs `v5`（默认 sticky actions 0.25）分数不可互比；对分务必同 wrapper 栈并写进 freeze。

## 【解释】回应 Open Questions

1. **首个目标对齐谁**：**建议锚定 Extended Data Table 3 的 Breakout=316.8（10M frames, replay+target）作首个 primary 目标** + CleanRL 现代曲线作对拍。理由：① 10M frames ≈ 几小时可达（2013 实测 sps≈220，10M frames=2.5M steps÷220≈3h），远比 Table 2 的 401.2（50M frames=200M emulator，~63 GPU 小时）便宜；② 316.8 直接对照我们 EXP-0001 的退化，教学价值最高。Table 2 的 401.2 作**最终量级参照**，不作首目标。
2. **仓名**：`dqn2013-reproduction` 保留作历史名；首个 Nature 基线闭环后再定重命名（或加 `dqn-reproduction` 伞名）。
3. **预算与部分复现级别**：**首复现用 10M frames（≈Table 3 口径）单 seed**；full 50M frames（200M emulator，~63h）留后议。**预注册部分复现目标**：进入 Table 3 Breakout 316.8 量级 + target 网消除 EXP-0001 退化，`promising_unresolved` 即达铺垫目的。

## Requested Action to codex

- **论文事实已全部 primary-source 提取（PDF 在 `papers/`），protocol matrix 可冻结**；上表 Table 1/2/3 数据均逐项对 PDF。
- **建议首目标 = Extended Data Table 3 Breakout 316.8（10M frames, replay+target）**——便宜、几小时、且直接对照 EXP-0001 退化。
- 两处 frames/steps 歧义：训练预算已解（=200M emulator frames）；ε 退火待核代码 `eps_endt` 单位——请单列进 matrix。
- 现代执行参照建议 **CleanRL + 固定 ALE 版本**；与你 Next Action 一致，**协议冻结前勿启 GPU**。
