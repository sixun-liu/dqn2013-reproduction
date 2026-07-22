# DQN 2013 到 Nature 2015 Breakout 复现

[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

本仓库提供一套可审计、可验证的 DQN Breakout 独立复现：先保留 2013 arXiv 版本的实现与失败分析，
再以 Nature 2015 Extended Data Table 3 的 replay + target 条件为主线完成单 seed 部分数值复现。

核心结果：现代 ALE 独立 PyTorch 实现在 10M agent decisions 后得到周期评估 peak/final mean
`350.18`，论文参考为 `316.8`。由于只有一个训练 seed、一个游戏且执行栈不同，结论限定为
**单任务、单 seed、现代执行栈下的部分数值复现**。

![Nature 2015 DQN Breakout reproduction](reports/assets/EXP-0004/nature2015_breakout_replication.png)

![Final Breakout policy](reports/assets/EXP-0004/dqn_breakout_demo.gif)

## 三分钟验证

Linux 和 Python 3.10/3.11 环境下：

```bash
git clone https://github.com/sixun-liu/dqn2013-reproduction.git
cd dqn2013-reproduction
DQN_ACCEPT_ROM_LICENSE=1 ./scripts/reproduce.sh setup
./scripts/reproduce.sh test
./scripts/reproduce.sh smoke
./scripts/reproduce.sh verify-reference
```

`setup` 会创建 CPU-only `.venv` 并安装验证所需依赖，不下载 CUDA wheel。Atari ROM 不在本仓库中；设置
`DQN_ACCEPT_ROM_LICENSE=1` 表示调用者已阅读并接受 AutoROM 提供的独立 ROM 许可证。

成功的 smoke 应完成：

- 256 agent decisions；
- 56 次 optimizer update；
- 4 次 Target Network 同步；
- held-out Q、独立 evaluation 和可加载 checkpoint；
- `verify_run.py` 输出 `"status": "ok"`。

更详细的安装、GPU、输出和故障排查见 [REPRODUCING.md](REPRODUCING.md)。

## 验证发布 checkpoint

`v0.1.0` Release 提供最终 checkpoint，不包含 ROM：

```bash
curl -L -o dqn2015-breakout-exp0004-s0-10m.pt \
  https://github.com/sixun-liu/dqn2013-reproduction/releases/download/v0.1.0/dqn2015-breakout-exp0004-s0-10m.pt
sha256sum -c reports/assets/EXP-0004/checkpoint.sha256
./scripts/reproduce.sh eval dqn2015-breakout-exp0004-s0-10m.pt
```

冻结评估协议为 epsilon `0.05`、seed `10000`、135K agent decisions、只统计完整 games。原运行和
独立复评都得到 60 局、mean `350.1833`，逐局 return 序列完全一致。跨硬件或依赖版本时应报告
实际结果，不把浮点级完全一致作为普适要求。

## 从头运行正式复现

```bash
DQN_ACCEPT_ROM_LICENSE=1 \
DQN_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  ./scripts/reproduce.sh setup-gpu
./scripts/reproduce.sh full
```

GPU 环境单独安装到 `.venv-gpu`。CUDA wheel 较大且与驱动/显卡有关，因此仓库不捆绑 CUDA，
也不会在默认验证路径中隐式下载它；上面的 `cu130` 是 EXP-0004 主机示例，其他主机应从
PyTorch 官方安装器选择匹配的 index。

正式配置为 [configs/public/nature2015_table3_10m.json](configs/public/nature2015_table3_10m.json)，
与冻结的 EXP-0004 formal config 除输出路径外逐字段相同。本机 RTX 5090 运行约 `7.93h`；ALE、
图像预处理和 replay 主要消耗 CPU/内存，因此 GPU 利用率呈脉冲状。

完整运行会输出：

```text
runs/<tag>/
├── .started
├── .completed
├── resolved_config.json
├── runtime.json
├── metrics.jsonl
├── heldout_states.npy
├── checkpoints.jsonl
└── checkpoints/
```

当前 checkpoint 保存网络、优化器和 RNG，但不保存 replay 与 ALE state，因此用于评估，不能声称
协议等价地恢复中断训练。

## 结果

| 项目 | 数值 |
|---|---:|
| 论文目标 | Nature 2015 Extended Data Table 3, Breakout replay + target |
| 论文参考 | 316.8 |
| 本地预算 | 10M agent decisions = 40M nominal emulator frames |
| 周期评估 | 40 次，合计 5,498 个完整 games |
| 本地 peak / final mean | 350.1833 |
| Final median | 373.5 |
| 独立 checkpoint 复评 | 60 games，mean 350.1833 |
| 训练墙钟时间 | 7.934 h |
| 裁决 | `promising_unresolved` |

机器可读结果位于 [reports/assets/EXP-0004](reports/assets/EXP-0004/)，运行：

```bash
./scripts/reproduce.sh verify-reference
```

即可检查所有公开结果文件的 SHA256、40 点曲线、headline 数值和 checkpoint 指纹。

CPU CI 配置保存在 [ci/github-actions-verify.yml](ci/github-actions-verify.yml)。当前发布凭据没有
GitHub `workflow` scope，因此它作为可直接启用的模板随仓库发布；使用具备该权限的凭据移动到
`.github/workflows/verify.yml` 后即可启用。

## 方法与实现

Nature 2015 执行器位于 [src/dqn2015_nature_breakout.py](src/dqn2015_nature_breakout.py)，包含：

- 三层卷积网络 `32/64/64 + FC512`；
- uniform replay，capacity 1M，50K warmup；
- 每 4 decisions 更新一次，batch 32；
- 每 10K decisions 硬同步 Target Network；
- centered RMSProp、clipped TD/Huber 梯度和 reward clipping；
- train/eval 分离的 life-loss terminal 与 full-game evaluation；
- JSONL、运行时版本、信标、held-out Q 和周期 checkpoint。

公开 JSON config 由 [scripts/run_nature2015_config.py](scripts/run_nature2015_config.py) 直接消费，
checkpoint 由 [scripts/evaluate_dqn2015_checkpoint.py](scripts/evaluate_dqn2015_checkpoint.py) 独立评估。
32 项单元测试覆盖网络尺寸、optimizer 一步公式、TD 梯度、epsilon、训练边界、Target Network、
wrapper、checkpoint、离线统计和发布配置 parity。

完整方法、代码走读、协议、结果与限制见
[DQN_REPRODUCTION_REPORT.md](reports/DQN_REPRODUCTION_REPORT.md)。

## 2013 与 2015 路线

| 项目 | 2013 arXiv 实现 | Nature 2015 主线 |
|---|---|---|
| 网络 | 两层卷积 + FC256 | 三层卷积 + FC512 |
| TD target | 当前 online Q | 延迟 Target Network |
| 主运行结果 | 峰值 10.90、最终 2.21，数值未复现 | peak/final 350.18，部分数值复现 |
| 用途 | 历史实现、失稳与单位问题分析 | 当前可执行、可验证主线 |

两条路线的架构、预处理、预算和评估不同，不能把它们当作 Target Network 的单变量因果消融。

## 代码谱系

- 论文事实：Mnih et al. 2013 与 Nature 2015 原文。
- 作者协议 oracle：DeepMind DQN 3.0 `9d9b1d1`，受限源码不在本仓库分发。
- 现代工程参照：CleanRL `fe8d8a0`，MIT attribution 见
  [LICENSES/CleanRL-MIT.txt](LICENSES/CleanRL-MIT.txt)。
- 本仓实现类别：`independent_reimplementation`。

协议恢复见 [references/NATURE2015_PROTOCOL_AUDIT.md](references/NATURE2015_PROTOCOL_AUDIT.md)，
第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

项目代码使用 [MIT License](LICENSE)。Atari ROM、原论文、DeepMind DQN 源码和 Release checkpoint
各自受其来源及适用条款约束；本仓库不分发 ROM 或 DeepMind 受限源码。
