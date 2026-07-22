# DQN 定量分析深化：离线数据 + 概率统计（Q 分布 / over-estimation）

> 2026-07-22，Claude。DQN 报告从"跑通 + 机制故事 + 可视化"再深化一层到**定量/数学分析**,
> 服务老师"把 RL 基础打扎实"。核心:**几乎不占大 GPU**(现有 checkpoint + 有限 rollout)。
> Claude 出机制/数学解读;数据提取、rollout、统计、画图归 codex(它握 run/checkpoint)。

## 一、目标
把"2013 Q 爆炸 vs 2015 稳定"从**现象**讲成**有数学根基的机制**,并给报告一层定量深度。

## 二、核心分析

### A. Q 值分布随训练阶段演化(直方图 / 分位数)
- 固定一批 **held-out 状态集**(全程复用,Nature held-out Q 口径),在 2013(EXP-0001)与 2015(EXP-0004)的**各阶段 checkpoint** 上算 Q 值分布。
- 看点:2013 是否**长尾/发散**(Q 爆到 61.74 的分布形态) vs 2015 是否**收紧稳定**(~3.72)。比只看 mean-max-Q 深一层。

### B. ★ Q over-estimation(高估偏差)量化 —— 数学深度最高
- **数学根基**:Q-learning 用 `max_a Q` 当 target,而 `E[max] ≥ max[E]`(Jensen / maximization bias)→ Q **系统性高估**;无 target network(2013)时高估**正反馈发散** → 就是 Q 爆炸。target network / Double DQN 正是来治它。
- **测法**(口径同 van Hasselt 2016):固定状态集,比 **Q 预测的价值 `Q_pred`** vs **从该状态用当前策略 rollout 到底的 Monte-Carlo 真回报 `G_MC`**,高估幅度 `gap = Q_pred − G_MC`,画 gap 随训练阶段、2013 vs 2015 的曲线。
- ⚠️ 口径诚实:`G_MC` 是"相对**当前策略**的回报"、非最优 `Q*`;测的是"Q 相对策略回报的高估",这是该领域标准近似,报告需标注。

### C. 附加(可选)
- TD error 分布随阶段(2013 是否发散);
- 状态/动作分布演化(探索→利用);
- 与 held-out max-Q 曲线(已有)、行为视频交叉印证。

## 三、数据与协议
- held-out 状态集:从某中期 checkpoint rollout 采一批状态,冻结、全程复用。
- checkpoint:2013(EXP-0001)、2015(EXP-0004)的周期 ckpt(**请确认各阶段 ckpt 是否留存**)。
- B 的 `G_MC`:每状态用对应 checkpoint 策略 rollout 到 episode 结束,Monte-Carlo 回报;有限条数即可,不占大 GPU。

## 四、Requested actions（给 codex）
1. 评估可行性 + 确认 2013/2015 各阶段 checkpoint 是否留存(不够则说明能到什么粒度)。
2. A:held-out 状态集上各阶段 Q 值分布(直方图/分位数数据)。
3. B:设计 over-estimation 量化协议并跑(`Q_pred` vs `G_MC` gap,2013 vs 2015 各阶段)。
4. 产出:Q 分布演化图 + over-est gap 曲线,收 `artifacts/dqn2013/analysis/`(或你习惯路径),给下载路径。

## 五、边界
单 seed、我们的实现;`G_MC` 为策略相关近似(非最优);结论作"机制理解 + 教学"用,不宣称与任何论文数值等同。

分工:Claude 数学/机制解读 + 讲解;codex 数据/rollout/统计/图;用户理解 + 复核。并行:Claude 已派子 agent 调研 DQN 可解释性文献(saliency/t-SNE/子策略等),可借鉴方法回来后并入本线。
