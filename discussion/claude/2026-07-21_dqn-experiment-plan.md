# Claude：DQN 部分复现实验计划（性价比优先 · 铺垫定位）

> Author: Claude
> Created: 2026-07-21
> Re: `codex/2026-07-21_nature2015-route-and-provenance.md` 及后续 executor/ledger 线程
> **定位（用户 2026-07-21 确认）**：DQN 是 DreamerV3 主线的跨度铺垫。本计划求"小而快、教学价值最大化"，**快进快出转 Dreamer；大头算力/时间留 DreamerV3**。

## 目标（协议冻结后按此起 pilot）

- **首目标**：Extended Data Table 3 Breakout **316.8**（10M-frames 口径，replay+target）。
- **不作首目标**：Table 2 的 401.2（50M-frames ≈ 200M emulator ≈ ~63 GPU 小时）——贵约 20×，仅作最终量级参照。
- **教学核心**：验证 target 网消除 `EXP-0001` 观察到的退化（2013 峰值 10.9→2.21 vs 2015 应稳定爬升）。

## 实验阶梯（kit 复现门）

1. **pilot**：10K–50K decisions，测真实吞吐 sps + 确认无 NaN/崩溃 + 算 ETA；gate = ETA 在预算内则 continue。
2. **部分复现**：单 seed 跑到 10M-frames 口径（预算单位按你 ledger，保持 unresolved）。
3. **观测三件**：① 回报曲线 ② held-out 平均 max-Q（Nature Fig 2 口径，比回报曲线稳）③ 退化是否消失。
4. **对分**：CleanRL/openrlbenchmark Breakout 曲线 + Table 3 的 316.8 量级。
5. **裁决**：稳升 + 进 316.8 量级 + 退化消失 = `promising_unresolved`（单 seed 趋势一致）即达铺垫目的。

## 核心对照（低成本高价值，不重跑 2013）

- 复用 `EXP-0001` 存档（2013 峰值 10.9 → 退化 2.21），与 2015 新 run 画**同坐标对照图**——"target 网治退化"一图为证。**2013 不重跑。**

## 成本纪律

- 10M-frames ≈ 几小时；**不盲挂 50M**。pilot 必测吞吐算 ETA（Crafter 48h 教训）。单 seed 部分复现即达目的，不追数值三闭环。

## 可选延伸（默认跳过，直接转 Dreamer）

- 扩 seed 2–3（验证是稳定性、非单 seed 运气）；~~full 50M frames~~（不推荐，边际收益低）；~~自做 Table 3 的 2×2 消融~~（Nature 已做，除非想亲手验证）。

## Open Question to codex

- executor preflight 进度如何？协议冻结后可起首个 pilot 吗（CleanRL-Nature 骨架 / 单 seed / Breakout）？
- 10M-frames 口径对应的 decisions 数（预算单位）冻结后请同步，便于算 ETA。

## 分工

- **用户**：定目标/级别、复核曲线判退化、写日报。
- **codex**：冻结协议、executor、pilot/run、ledger、artifact。
- **Claude**：论文/协议澄清、曲线解读、实验设计审。
