# Reproducing and Verifying the DQN Breakout Result

本文档区分四个验证层级，避免把“测试通过”“smoke 跑通”和“论文结果复现”混为一谈。

| 层级 | 命令 | 用时参考 | 证明什么 |
|---|---|---:|---|
| 单元测试 | `reproduce.sh test` | 秒级 | 公式、边界、wrapper 和分析函数符合代码契约 |
| CPU smoke | `reproduce.sh smoke` | 秒至分钟 | 环境、replay、训练、target sync、评估、checkpoint 全链路可运行 |
| Checkpoint 复评 | `reproduce.sh eval` | 本机约 2.6 分钟 | 发布权重在冻结评估协议下产生可用策略与分数 |
| 从头复现 | `reproduce.sh full` | 本机约 7.93 小时 | 重新训练 10M decisions 并生成完整 40 点曲线 |

## 1. 系统要求

推荐环境：

- Linux x86_64；
- Python 3.10 或 3.11；
- CPU smoke：8 GB 以上内存；
- 正式训练：NVIDIA GPU、可用 CUDA PyTorch、足够存储 checkpoints；
- Atari Breakout ROM，由用户通过 AutoROM 独立安装并接受其许可证。

EXP-0004 的已验证环境为 Python 3.10.19、PyTorch 2.13.0+cu130、Gymnasium 0.29.1、ALE-Py
0.8.1、Stable-Baselines3 2.3.2。完整版本表见
[requirements/verified.txt](requirements/verified.txt)。

## 2. 安装

```bash
git clone https://github.com/sixun-liu/dqn2013-reproduction.git
cd dqn2013-reproduction
DQN_ACCEPT_ROM_LICENSE=1 ./scripts/reproduce.sh setup
```

脚本创建 CPU-only `.venv`，先从 PyTorch 官方 CPU wheel index 安装 PyTorch，再执行 editable
install 和 AutoROM。该默认路径不会下载 CUDA runtime。仓库不会自动代表用户接受 ROM 许可证，
没有显式设置 `DQN_ACCEPT_ROM_LICENSE=1` 时会停止。

若已经有兼容环境，可避免新建 venv：

```bash
export DQN_PYTHON=/path/to/python
./scripts/reproduce.sh test
```

CUDA build 与显卡驱动相关，因此不属于默认验证安装。`requirements/verified.txt` 只记录产生
EXP-0004 的完整版本，不应直接当作跨机器安装命令。

## 3. 单元测试

```bash
./scripts/reproduce.sh test
```

发布版包含 32 项测试，至少覆盖：

- Nature 网络输出尺寸和 `1,686,180` 参数量；
- centered RMSProp 一步更新公式；
- clipped TD/Huber 梯度；
- warmup、epsilon、update 和 target-sync 边界；
- train/eval life-loss 与 reward clipping wrapper；
- checkpoint 内容与中断信号；
- public full config 与冻结 formal config 的逐字段 parity；
- Q/FC512、bootstrap 和视觉干预函数。

## 4. CPU smoke

```bash
./scripts/reproduce.sh smoke
```

也可以指定输出目录：

```bash
./scripts/reproduce.sh smoke /tmp/dqn-smoke
```

配置见 [configs/public/nature2015_smoke_cpu.json](configs/public/nature2015_smoke_cpu.json)。它只运行
256 decisions，不形成性能结论。`verify_run.py` 要求以下 completion signals 同时存在：

- `.started` 与 `.completed`；
- resolved config、runtime 和 JSONL metrics；
- optimizer update 与 Target Network sync 均实际发生；
- held-out state 捕获和有限 Q 诊断；
- Nature 网络参数量、Breakout action space 与 wrapper stack 正确。

重复使用同一输出目录会因 `.started` 的排他创建而失败，这是防止误启动双进程的预期行为。

## 5. 验证已提交结果

```bash
./scripts/reproduce.sh verify-reference
```

该命令读取 [reports/assets/EXP-0004/manifest.json](reports/assets/EXP-0004/manifest.json)，检查：

- 五个公开结果/展示文件的 SHA256；
- evaluation CSV 恰好包含 40 个点；
- final mean 为 `350.18333333333334`；
- 论文参考为 `316.8`；
- 独立复评逐 episode 完全一致；
- final checkpoint SHA256 为
  `73e3e71f437bf07f59128b712f8a7e294c23052b9d6d5c62cb2478b58d672ef0`。

这验证已发布证据包内部一致，不等于重新训练模型。

## 6. 下载和评估 checkpoint

```bash
curl -L -o dqn2015-breakout-exp0004-s0-10m.pt \
  https://github.com/sixun-liu/dqn2013-reproduction/releases/download/v0.1.0/dqn2015-breakout-exp0004-s0-10m.pt
sha256sum -c reports/assets/EXP-0004/checkpoint.sha256
./scripts/reproduce.sh eval dqn2015-breakout-exp0004-s0-10m.pt
```

默认使用 CPU，避免为了复评 checkpoint 安装数 GB 的 CUDA runtime。已有 GPU 环境时可显式启用：

```bash
DQN_PYTHON=.venv-gpu/bin/python DQN_EVAL_DEVICE=cuda \
  ./scripts/reproduce.sh eval dqn2015-breakout-exp0004-s0-10m.pt
```

评估器固定：

- `BreakoutNoFrameskip-v4`，sticky=false；
- epsilon `0.05`；
- eval seed `10000`；
- 135K agent decisions；
- 不把 life loss 当 terminal；
- 不裁剪评估 reward；
- 只统计窗口内完成的完整 games。

原运行与独立复评均得到 60 games、mean `350.1833`、median `373.5`。同版本与硬件应高度接近；
跨版本时报告差异，不应为匹配单个数字替换 seed 或 checkpoint。

## 7. 从头运行 10M-decision 复现

先从 PyTorch 官方安装器选择与本机驱动匹配的 CUDA wheel index，再显式创建独立 GPU 环境：

```bash
DQN_ACCEPT_ROM_LICENSE=1 \
DQN_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  ./scripts/reproduce.sh setup-gpu
```

`cu130` 是 EXP-0004 主机所用版本的示例，其他主机应替换成 PyTorch 官方安装器给出的 index。
这一步会下载较大的 GPU 依赖并创建 `.venv-gpu`；只有从头训练需要它。仓库本身不保存 CUDA、
PyTorch wheel、ROM 或训练 checkpoint。裸 `pip install -e .` 也不会安装运行时依赖；这是有意设计，
请使用上述 setup 命令，或在自管环境中显式安装 `.[runtime,atari]`。

```bash
./scripts/reproduce.sh full
```

指定输出目录：

```bash
./scripts/reproduce.sh full /data/runs/dqn-nature-table3
```

public full config 与历史 frozen config 仅 `output_dir` 不同，并由测试锁定。正式运行应产生 40 次
135K-decision 周期评估和最终 `.completed`。运行结束后脚本自动执行：

```bash
python scripts/verify_run.py --run-dir <output> --mode full
```

本机实测：

- 10,000,000 agent decisions；
- 40,000,000 nominal emulator frames；
- 2,487,500 optimizer updates；
- 7.934 h 墙钟时间；
- 40 次评估，共 5,498 个完整 games。

DQN 的环境推进、图像预处理和 replay 主要在 CPU；GPU 每四个 decisions 执行一次 batch 32 小网络
更新，利用率通常呈脉冲状。

## 8. 输出与恢复边界

每个 run 保存完整 expanded config、环境/依赖版本、wrapper 列表、JSONL metrics、held-out states、
checkpoint 索引和信标。checkpoint 包含：

- online/target network；
- optimizer state；
- Python、NumPy、PyTorch RNG state；
- 完成步数和 expanded args。

它不包含 replay buffer 或 ALE state。因此 checkpoint 可以用于独立评估和网络分析，但不能在中断
后声称协议等价地恢复训练。

## 9. 常见问题

### `Namespace ALE/Breakout-v5 not found` 或 ROM 错误

确认已经显式运行：

```bash
DQN_ACCEPT_ROM_LICENSE=1 ./scripts/reproduce.sh setup
```

并在当前 Python 环境中检查：

```bash
python -c "import gymnasium as gym; gym.make('BreakoutNoFrameskip-v4').close()"
```

### CUDA 不可用

Smoke 和默认 checkpoint 复评均使用 CPU。正式训练需要先执行显式 `setup-gpu`；launcher 在
`--device cuda` 不可用时会直接失败，不会静默退回 CPU 长跑。

### 输出目录已经存在

运行器使用排他 `.started` 信标。换一个新目录；不要删除历史信标后原地重跑。

### 为什么没有 Docker 镜像

GPU driver、CUDA wheel 和 Atari ROM 许可证仍需要主机侧处理。当前 venv + pinned stack + CI 比一个
暗含驱动和 ROM 的镜像更透明。后续可增加不含 ROM 的 CPU validation image，但不是复现必要条件。

## 10. 证据边界

CPU CI 模板位于 [ci/github-actions-verify.yml](ci/github-actions-verify.yml)，包含依赖安装、AutoROM、
unit tests、reference verification 和 smoke。当前发布凭据缺少 GitHub `workflow` scope，因此模板
尚未激活为 `.github/workflows/`；本地一键验证不受影响。

本项目是 `independent_reimplementation`，不是作者原始代码复跑。`350.18` 支持 Breakout 单 seed
进入论文 Table 3 分数量级，不支持完整复现 49 个游戏、多 seed 统计等价或 Target Network 的本地
单因素因果结论。详细说明见 [复现报告](reports/DQN_REPRODUCTION_REPORT.md)。
