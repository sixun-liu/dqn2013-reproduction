# DQN 评估协议参考(开源 wrapper)· 解 collect_calibration 两个失败

> 2026-07-22,Claude。codex `collect_dqn2015_calibration_trajectories.py` 两失败
> (`noop31`: post-FIRE no-op 到 terminal;`cap135k`: 135K decisions 内只 15/23 games)。
> 根因 = **自写 no-op/FIRE burn-in,没照标准 Atari wrapper 处理 terminal 重采**。
> Claude 已下载 CleanRL + SB3 扫读,开源早已标准处理。**不甩链接——下面是确切解法 + 行号。**

## 开源代码位置(已双存)
- 本地 `/root/lsx/Research/DreamerV3/references/dqn_openref/{sb3_atari_wrappers.py, cleanrl_dqn_atari.py}`
- 源:`github.com/DLR-RM/stable-baselines3` common/atari_wrappers.py · `github.com/vwxyzjn/cleanrl` cleanrl/dqn_atari.py(用 `cleanrl_utils/atari_wrappers.py`,与 SB3 逐条一致)

## 解 `noop31`(no-op/FIRE burn-in 触发 terminal)
参照 `sb3_atari_wrappers.py`:
- **`NoopResetEnv.reset`(L59–72)**:no-op 数 = `np_random.integers(1, noop_max+1)`(**随机 1~30、不固定 31**);循环里 `if terminated or truncated: obs,info = env.reset()` —— **遇 terminal 就 reset 重采、不 raise**。
- **`FireResetEnv.reset`(L87–95)**:FIRE(step 1)与 step 2 后各 `if terminated or truncated: env.reset()`。
- **`EpisodicLifeEnv.reset`(L142–154)** 注释逐字命中:"the no-op step **can lead to a game over**, so we need to check it again...to avoid `RuntimeError`"。
→ codex 的 `post_fire_noop_burnin` 把 "遇 terminal" 改成 **reset 重采**(而非 `raise RuntimeError`)即解。

## 解 `cap135k`(预算内凑不够 games)
- **budget(decisions) 与 games 数别同时硬卡**:要么跑满 budget、有几局算几局;要么按局数、不设 decisions 上限。
- 用 `EpisodicLifeEnv`(life-loss = terminal)让"局"更短,更易在预算内凑够统计局数。

## 标准 wrapper 栈(Nature 口径,`AtariWrapper` L289–314)
`NoopReset(noop_max=30) → MaxAndSkip(skip=4) → EpisodicLife → FireReset → WarpFrame(84×84 gray) → ClipReward`;`action_repeat_probability=0`(sticky off);env `BreakoutNoFrameskip-v4`;gamma 0.99。CleanRL 同套。

## ★ 关键:这套正好满足 codex 要的"训练一致语义"
codex 坚持 calibration 轨迹用训练同语义(clipped reward / life-loss terminal / γ=0.99)——**标准 wrapper 恰好全给了**:`ClipRewardEnv`=clipped reward、`EpisodicLifeEnv`=life-loss terminal、γ 在 agent 侧。**直接复用 cleanrl/sb3 的 atari_wrappers 比自写更对齐、更省**,还免掉 burn-in 的 terminal 坑。

## Requested action(给 codex)
1. `collect` 脚本的 no-op/FIRE burn-in 改成"遇 terminal reset 重采"(照 `NoopResetEnv`/`FireResetEnv`),或直接复用标准 atari_wrappers;
2. decisions budget 与 games 数解耦。

分工:Claude 开源考据 + 解法;codex 改脚本 + 重跑采集。
