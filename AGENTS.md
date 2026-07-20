# AGENTS.md - DQN 2013 到 Nature 2015 单任务复现契约

## 仓库边界

- 本目录是 control repo，同时跟踪项目独立实现 `src/`；角色真源见 `research/repositories.yaml`。
- CleanRL、`deep_q_rl` 和 DeepMind DQN 是独立 third-party repo，只固定 remote/commit/license。
- run、checkpoint、artifact、ROM 和论文 PDF 位于 Git 外的数据盘。
- `EXP-0001`--`EXP-0003` 早于首次 Git 提交；历史证据以 freeze、文件 SHA 和 registry 为准。

## 目标

- `EXP-0001`--`EXP-0003` 保留为 arXiv `1312.5602` Breakout 独立重实现的历史与机制证据，
  不继续投入约 5M-update 长程扩展。
- 当前主线是理解并跑通 Nature 2015 DQN 的 Breakout 结果。DeepMind DQN 3.0 只读恢复作者协议，
  现代可执行实现必须单独记录 lineage 和协议漂移。
- 单任务、单 seed 起步；不宣称复现全部 49 个 Atari 游戏或多 seed 数值。

## 运行纪律

1. 进程控制权归 Codex；启动使用 detached 进程、绝对输出路径和 `.started` 信标。
2. GPU 判忙只看 `nvidia-smi --query-compute-apps=pid`。
3. run 必须保存 freeze、stdout、JSONL 指标和模型 checkpoint。
4. 大文件只写 `/root/autodl-tmp/`；不得写入系统盘缓存。
5. 观察与解释分开。现代 ALE 或运行时漂移必须显式记录。

## 证据权限

- 2013 与 Nature 2015 论文 PDF/hash 是 primary source；Nature 论文尚未冻结前不得进入正式复现。
- DeepMind DQN 3.0 commit `9d9b1d13a2b491d6ebd4d046740c511c662bbe0f` 是 Nature 2015
  author code，许可仅限学术审查，不得复制进公开控制仓。
- CleanRL commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` 是 third-party reference。
- `src/dqn2013_breakout.py` 是 independent reimplementation。
- 单 seed 趋势最多裁决为 `promising_unresolved`。

## Discussion 协作

- `discussion/claude/`：Claude 写研读、来源摘录和独立复核；Codex 只读。
- `discussion/codex/`：Codex 写服务器审计、协议问题和运行观察；Claude 只读。
- `discussion/INDEX.md`：Codex 维护未决线程路由，不保存 canonical state。
- 经核验事实进入 `references/`；持久决策进入 `DEVLOG.md`；实验状态和证据进入 `research/`。
- discussion 内容本身不能支持论文复现 claim，服务器进程控制权仍归 Codex。

## 精简执行方式（用户 2026-07-21 批准）

1. 同一目标主张、代码谱系和协议路线只创建一个 replication cycle；smoke、pilot 和正式 run 用不同
   run tag 与 resolved config 区分，不为每个阶段重复建 EXP。若算法行为、目标协议或 executor 改变，
   必须重新 freeze，必要时新建 cycle。
2. 计算前的 clean/pushed commit、`new + freeze`、expanded config、seed/预算单位、绝对 output、
   started 信标、GPU 判重、停止条件和成本门不可省略。
3. 运行中只维护原始日志、run STATUS、必要的模型 checkpoint 和关键边界 checkpoint event；不因
   普通进度机械重写 `CURRENT_STATE/PLAN/TODO/DEVLOG`。
4. smoke/pilot 只证明工程与成本，不进 scoreboard、不形成论文 claim，也不要求逐次人工 review。
5. 结案默认只生成一份同坐标审查图，执行一次 observation/artifact/close；仅当方向、正式裁决或近期
   动作变化时更新相应热文档。
6. scheduler 保持只读且默认不 enqueue；discussion 只为真实协议冲突或跨 agent 交接新建文件。
7. 用户关注点集中在正式运行成本门和最终对照图。只有安全、预算、权限边界或目标主张发生实质变化
   时中断长期作业请求用户裁决。
