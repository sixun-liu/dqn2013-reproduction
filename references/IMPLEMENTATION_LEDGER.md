# DQN 实现谱系 Ledger

| ID | Stable path | Commit / hash | License | Source class | 用途 |
|---|---|---|---|---|---|
| P1 | `/root/autodl-tmp/papers/dqn_1312.5602.pdf` | SHA256 `8db04120cace173151c77e0faa6f3eaa4207009da66b9417597dc70bfee56d9c` | paper | primary | 2013 主张与公开协议真值 |
| P2 | `/root/autodl-tmp/papers/nature14236.pdf` | SHA256 `cc811007a48aea14fcc135158ed96d01982930f415045a19f89474bfa1a74eb5` | paper | primary | Nature 2015 主张与公开协议真值 |
| A1 | `/root/autodl-tmp/third_party/deepmind-dqn` | commit `9d9b1d13a2b491d6ebd4d046740c511c662bbe0f` | Limited academic review license | author | 2015 作者代码谱系与 step/frame 语义；不可当 2013 原码 |
| T1 | `/root/autodl-tmp/third_party/deep_q_rl` | commit `34202491f5d8c219f52410d19cd0e91d44c37ecc` | BSD-3-Clause | third_party | 2013/NIPS 参数缺口的交叉核对 |
| T2 | `/root/autodl-tmp/third_party/cleanrl` | commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` | MIT | third_party | 现代工程结构与 2015 差异说明 |
| L1 | `/root/autodl-tmp/dqn-reproduction/src/dqn2013_breakout.py` | SHA256 `77aa269cd39888ebae4b0c256a120056141f96978284428801f2493e23ee62c0` | project-local | independent_reimplementation | EXP-0001 冻结实现 |

## 关键文件 hash

- A1 `README.md`: `ad76e550b42bded20fe9b17e854203427c13f8df1c97f201a679efee637dd6e5`
- A1 `dqn/LICENSE`: `d81b5b5bf24eb99e73b067cb4f8bf88e16e00dd23de80f5fbec34eb3eb68a5b9`
- A1 `run_gpu`: `5c9c6ec0439016f6d7539a29a1c9b823c2548f23cdaa491932690976ad9c08a1`
- A1 `NeuralQLearner.lua`: `c6633fe5fee67f3366793cfe48c379db9743369e12d96c64779db5cd0d20941d`
- A1 `convnet_atari3.lua`: `7cb610fdc33327b78660f93018d4d6ef89491037c63af33e984b5e3e0e5f4187`
- T1 `README.md`: `3850e8bc13c112d862839ab4d77b4b8fbafe452b3484091a009f1972ee3d8001`
- T1 `LICENSE.txt`: `307464d74dc0ecd4a544a2b20e480f5b64d2eb8d846c24e5eb1d8aa338494110`
- T1 `run_nips.py`: `1e5fe09ed215916708a1cd1783fc6898ebc82f5bde5ed32c5f9b9c2c1d41c90a`
- T2 `cleanrl/dqn_atari.py`: `84ec363765bf3493761186eb1c7ea7ae7dcadaebed3ddddebdf4479bd2dd34f2`

A1 的许可限制商业使用、复制和分发，本项目只保留原仓 clone 用于学术核对；不得把其源码复制进
本地实现。T1 和 T2 的结论仍是第三方证据，遇到与对应 primary paper（P1/P2）冲突时不晋升为
论文事实。

## Nature 2015 执行实现差异（代码事实）

对照版本固定为 A1 `9d9b1d1` 与 T2 `fe8d8a0`。论文事实仍以 P2 为准；下表只描述已直接核验的
代码行为，不用“Nature-style”名称替代逐项协议核对。

| 字段 | DeepMind DQN 3.0（A1） | CleanRL `dqn_atari.py`（T2） | 本项目处理 |
|---|---|---|---|
| 谱系与许可 | author code；limited academic review | third-party；MIT | A1 只读作协议 oracle；T2 作现代 executor scaffold |
| 运行栈 | LuaJIT、Torch7、CUTorch、Xitari、AleWrap | Python、PyTorch、Gymnasium、ALE-Py | 当前机器不部署 A1 老栈；执行代码从 T2 衍生并保留 MIT attribution |
| 网络 | 32@8/4、64@4/2、64@3/1、FC512 | 相同三卷积和 FC512 | 结构可直接沿用 T2，并做参数量/前向 parity check |
| replay/update | capacity 1M、warmup 50K、batch32、每4 decisions 更新 | capacity 1M、warmup 80K、batch32、每4 decisions 更新 | warmup 改为 50K；replay 内存与恢复策略单独冻结 |
| target network | hard sync 每 10K decisions | hard sync 默认每 1K decisions | 改为 10K，并用单元测试核对更新边界 |
| 优化与 TD loss | 自定义 centered RMSProp，lr 2.5e-4；TD error clip ±1 | Adam，lr 1e-4；MSE | 不能只改 CLI；实现 A1/P2 对齐的 RMSProp 与 clipped-TD/Huber 语义 |
| 探索 | 1.0 -> 0.1，`eps_endt=1M` | 1.0 -> 0.01，默认用总预算 10% | 终值改 0.1；退火计数单位在 freeze 前继续对账 |
| 训练预算 | `steps=50M`、action repeat 4；代码打印 `frames=steps*4` | 默认 10M timesteps、MaxAndSkip 4 | 配置变量显式使用 `agent_decisions` 与 `emulator_frames`，不沿用孤立 `frames` |
| 环境/预处理 | random starts 30、相邻两帧 max、repeat4、84x84 luminance | Noop30、MaxAndSkip4、EpisodicLife、resize/grayscale、stack4 | 固定 ALE v4 sticky=false；逐项核对 wrapper 顺序和 train/eval life-loss 语义 |
| 评估 | 每250K decisions，125K decisions，epsilon=.05；测试不因丢命结束 | 默认无周期独立评估；保存后评10 episodes，epsilon=`end_e` | 新增独立 Nature evaluator，不使用训练 return 或 CleanRL 默认 final eval 冒充 |
| 保存/恢复 | 周期 `.t7` 保存 | 默认仅可选保存最终模型，无周期 resume | 增加绝对输出、JSONL、信标、优雅停止和模型 checkpoint；replay 恢复权限单列 |

### 当前采用决定

采用“P2/A1 定义协议，T2 提供现代执行骨架”的组合。后续衍生 executor 归类为
`independent_reimplementation`，不得声称运行官方代码。A1 源码不得复制进公开控制仓。

### 尚未冻结的两个解释

1. P2 Extended Data Table 3 的 `10M frames` 不能先换算为 2.5M decisions。A1 的计数器以 decision
   递增，同时把 `steps * actrep` 打印为 emulator frames；Table 3 对应 10M decisions 还是 10M
   emulator frames，必须结合 P2 原文术语与作者实现再裁决。
2. `replay,no-target` 是 Nature 协议内部的 target-network 消融，只能支持 target 机制比较；它不等同
   于完整 2013 DQN，也不能直接与 EXP-0001 作数值等价比较。
