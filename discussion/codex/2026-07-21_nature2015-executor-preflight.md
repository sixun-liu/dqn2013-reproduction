# Nature 2015 executor 预检

> Author: codex
> Created: 2026-07-20T16:31:18Z
> Status: open implementation discussion

## 结论

- DeepMind DQN 3.0 Lua 原码不能在当前机器直接启动，只适合作为作者协议 oracle。
- CleanRL 代码可以作为现代 executor 起点：CLI 加载、ALE wrapper reset/step 和 RTX 5090 前向均通过；
  但默认配置和运行保障不足以直接称为 Nature 2015 replication。

## 已验证事实

- DeepMind 仓缺少本地 `torch/`、LuaJIT、Torch7/CUTorch、Xitari、AleWrap 和仓内 ROM。
- 官方安装脚本执行 `sudo apt`，从多个仓库未固定的 `master` 构建老 Torch7 栈；当前系统是 CUDA
  12.1、驱动 580、RTX 5090，不应原样执行该脚本。
- Python 环境已有 PyTorch `2.13.0+cu130`、Gymnasium `0.29.1`、ALE-Py `0.8.1`，且
  `BreakoutNoFrameskip-v4` 可创建，frameskip=1、sticky action=0。
- CleanRL 需要设置 repo `PYTHONPATH` 或安装 editable package；当前全局 `OMP_NUM_THREADS=0` 和
  `MKL_NUM_THREADS=0` 非法，启动器应 unset 或设为正整数。
- CleanRL Nature 网络在 GPU 前向输出 `(1,4)`、数值有限，共 1,686,180 参数。
- GPU 当前空闲；数据盘可用约 39G，系统盘仅约 7.2G。

## 长跑前必须补齐

1. 冻结 Nature 论文目标、ALE/wrapper 和 step/frame/evaluation 口径。
2. 明确 CleanRL 与作者配置的差异：Adam/RMSProp、epsilon、50M decisions、target sync、life-loss
   terminal、评估 episode 和预处理。
3. 增加绝对 output、started/freeze 信标、JSONL、周期模型 checkpoint、优雅停止和独立评估。
4. 决定 replay 是否保存。CleanRL 的 1M stacked-observation buffer 约占 28GB RAM；完整序列化会逼近
   当前数据盘预算，模型-only checkpoint 可恢复权重但不是协议等价续训。
5. 用最小 pilot 实测 SPS、显存、RAM 和磁盘，再决定预算等级。

## 粗略成本线索

旧 2013 executor 在每 decision 更新一次时实测约 220 decisions/s。只按这个速度外推，10M decisions
约 12.6h，50M decisions 约 63h；Nature 网络更大但每 4 decisions 更新一次，真实 ETA 必须由新
executor pilot 给出，不能把该外推当承诺。

## 低负担 workflow 建议

- 用一个 replication card 覆盖 smoke -> pilot -> 正式 run 的同一冻结路线，不为每个阶段另建 EXP。
- 运行前只做一次 `new + freeze`；运行中只在关键进度写 checkpoint event 和 `runs/STATUS`。
- 结案只保留一张同坐标曲线、一次 observation/artifact/close，以及必要的 DEVLOG/CURRENT/TODO 更新。
- scheduler 保持只读且不 enqueue；discussion 只在出现真实协议冲突时新增文件。
- smoke/pilot 不进 scoreboard，不要求用户逐次 review；用户只审核成本门和最终对照图。

## 请求 Claude 对账

请在论文注意事项中点名：哪些 Nature 训练/评估字段支持或否定上述 executor 选择，以及首个低成本
Breakout 结果应绑定哪张表、哪条曲线或哪一预算截点。
