# DQN 可解释性分析 · 方法实现规格（给 codex）

> 2026-07-22,Claude。给 DQN 报告的"理解/可解释"深化章节。**Claude 已研读一手论文**
> (van Hasselt/Greydanus/Anand 三篇精读方法节,t-SNE/梯度saliency/caveat 为成熟方法),
> 提取"方法 → 我们怎么在 checkpoint 上落地"规格。**不甩链接给 codex,这份是消化后的实现规格。**
> 论文 PDF 已双存档(hash 见文末),子 agent 检索地图见 `docs_from_claude/`(或我转达)。

**前提**:全部**复用现有 DQN checkpoint**(Nature 84×84×4,3 卷积 32/64/64 + FC512),几乎不占 GPU;**2013-塌缩 vs 2015-稳定 对比贯穿全部**。

## 方法1 · Q over-estimation 追踪(van Hasselt 2016,★最优先)
- **协议**:`value estimate` = 定期在评估阶段访问状态上算 `mean_t max_a Q(sₜ;θ)`;`ground truth` = 用**该 checkpoint 策略**从这些状态 rollout 到底、算实际**折扣回报**(Monte-Carlo,γ=0.99);`gap = est − truth`。
- **★ 招牌**:复现该文 Fig 3——**2013 的 Q 估计 log-scale 爆炸**(我们是 61.74)且**爆炸时间点与 score 塌缩精确吻合** = "Q 爆炸是因、回报崩是果"。
- **可选加分消融**:改 Double-DQN 目标 `Y=R+γQ(s′,argmaxₐQ(s′,a;θ);θ⁻)`,看能否缓解 2013 塌缩。
- 成本:低。(与已发 `2026-07-22_dqn-offline-quant-analysis.md` 合并;本条补 van Hasselt 确切协议。)

## 方法2 · 显著性 saliency(Greydanus 2018 扰动 + Wang 2016 梯度)
- **扰动式(主)**:`Φ = I⊙(1−M) + blur(I)⊙M`,M=以(i,j)为心的高斯 mask(σ²=25),blur σ=3,每 **k=5 像素**采一点再上采样;`S_Q(i,j)=½‖Q(I)−Q(Φ)‖²`(对 Q 向量或选中动作)。
- **梯度式(快速对照)**:`s.requires_grad_() → Q.max().backward() → |s.grad|`。
- **★ 招牌**:Breakout **tunneling**——agent 挖隧道时 saliency 指向砖墙隧道口(复现该文 Fig 3e,f);叠 2013 vs 2015(塌缩 agent 预期不再跟球)。
- ⚠️ **必配方法5 的 caveat**。成本:低–中。

## 方法3 · t-SNE 表征可视化(Zahavy 2016 + Mnih 2015 Nature)
- 采一整段 episode 的 **512-d FC 激活** → `sklearn.manifold.TSNE` 降 2D → 按 Q值/分数/剩余生命/时间步上色。
- **★ 招牌**:复刻 Nature Fig 4(相似价值状态聚簇);2013-塌缩 vs 2015-稳定 的簇结构对比(塌缩预期簇崩坏)。成本:低(CPU 分钟级)。

## 方法4 · AtariARI 线性探针(Anand 2019,量化候补——比定性图硬)
- AtariARI gym wrapper 采 (frame, RAM label) 对(Breakout 35 标签:`ball_x`/`ball_y`/`player_x`+每块砖存在);冻结 checkpoint **512-d 特征**训**线性探针**预测这些变量、报 **F1**(坐标分 bin 分类)。
- **量化对比**:2013 vs 2015 探针 F1(塌缩 agent 是否丢失"球位置"可解码性)。
- **gotcha**:标签由 RAM 提取(非像素),与 84×84 下采样输入无坐标冲突;用 `AtariARIWrapper` 逐帧对齐,别拿下采样图反推标签坐标尺度。成本:低–中(+接 AtariARI 依赖)。

## 方法5 · ⚠️ saliency 方法论 caveat(Atrey 2020 ICLR,必带)
RL 显著图易过度解读、**"探索工具非解释工具"**;放 saliency 图**必须配反事实验证**(手动挪走球/砖,看 Q 与 saliency 是否如预期变化)。报告里作研究严谨性教学点。

## 暂不做(需重训/工程量高)
Zahavy SAMDP option 发现、Mott 2019 attention agent、Juozapaitis 2019 reward decomposition——报告作"进一步方向"引用即可。

## 优先级 & Requested actions(给 codex)
优先级(性价比+教学+不占算力):**① over-est ② t-SNE ③ saliency(+caveat) ④ AtariARI 探针**。
1. over-est:按 van Hasselt 协议出 2013 vs 2015 的 Q-est vs MC-return gap 曲线 + log-scale Q 爆炸图。
2. t-SNE:512-d 特征 → TSNE → 上色,2013 vs 2015 对比。
3. saliency:2013/2015 checkpoint 的 Breakout saliency(扰动为主、梯度对照),含 tunneling 帧 + 一例 Atrey 反事实验证。
4. (量化候补)AtariARI 探针 F1,2013 vs 2015。
5. 产出收 `artifacts/dqn2013/interpretability/`,给下载路径。

## 论文存档 & 原文/代码链接(方便 codex 查原文、参考现成实现)

**PDF 已双存档,codex 可直接读服务器上的 PDF、无需下载**:
- 服务器 `/root/autodl-tmp/papers/interpretability/<id>.pdf`(codex 直接 Read)
- 本地 `/root/lsx/Research/DreamerV3/references/interpretability/<id>.pdf`
- arXiv 网页/PDF = `https://arxiv.org/abs/<id>` / `https://arxiv.org/pdf/<id>`

| 论文 | arXiv id | ★ 官方代码仓(实现优先参考) | sha256前16 |
|---|---|---|---|
| van Hasselt 2016 over-est | 1509.06461 | 无官方码(方法简单,按 spec 方法1) | a37336d7 |
| Greydanus 2018 saliency | 1711.00138 | github.com/greydanus/visualize_atari | 85449caa |
| Anand 2019 AtariARI 探针 | 1906.08226 | github.com/mila-iqia/atari-representation-learning | 5ba7be92 |
| Atrey 2020 saliency caveat | 1912.05743 | github.com/KDL-umass/saliency_maps | 6bcbc1af |
| Zahavy 2016 t-SNE | 1602.02658 | t-SNE 用 sklearn(按 spec 方法3) | 176d8921 |
| Wang 2016 Dueling 梯度saliency | 1511.06581 | 一行 backward(按 spec 方法2) | 11239e34 |

**★ 实现请优先参考官方代码仓**(比从论文重写省事且更准):`visualize_atari` 有扰动 saliency 现成代码;AtariARI repo 有 Breakout RAM wrapper(`ram_annotations.py`)+ 线性探针基准;`saliency_maps` 有反事实验证实现。GitHub 拉取走服务器代理(同下 arxiv 那条链路)。

分工:Claude 研读/方法规格/机制解读;codex 实现/画图;用户审 + 讲。
