# AGENTS.md - 2013 DQN 单任务复现契约

## 仓库边界

- 本目录是 control repo，同时跟踪项目独立实现 `src/`；角色真源见 `research/repositories.yaml`。
- CleanRL、`deep_q_rl` 和 DeepMind DQN 是独立 third-party repo，只固定 remote/commit/license。
- run、checkpoint、artifact、ROM 和论文 PDF 位于 Git 外的数据盘。
- `EXP-0001`--`EXP-0003` 早于首次 Git 提交；历史证据以 freeze、文件 SHA 和 registry 为准。

## 目标

- 理解并部分复现 arXiv `1312.5602` 的 Breakout 结果。
- 单任务、单 seed；不宣称复现全部七个 Atari 游戏或多 seed 数值。
- CleanRL 只作第三方工程参照，不能把 2015 Nature DQN 当成 2013 论文实现。

## 运行纪律

1. 进程控制权归 Codex；启动使用 detached 进程、绝对输出路径和 `.started` 信标。
2. GPU 判忙只看 `nvidia-smi --query-compute-apps=pid`。
3. run 必须保存 freeze、stdout、JSONL 指标和模型 checkpoint。
4. 大文件只写 `/root/autodl-tmp/`；不得写入系统盘缓存。
5. 观察与解释分开。现代 ALE 或运行时漂移必须显式记录。

## 证据权限

- 论文 PDF/hash 是 primary source。
- CleanRL commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` 是 third-party reference。
- `src/dqn2013_breakout.py` 是 independent reimplementation。
- 单 seed 趋势最多裁决为 `promising_unresolved`。
