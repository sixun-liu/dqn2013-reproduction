# 2013 DQN Breakout 部分复现

本项目以 Mnih et al. 2013《Playing Atari with Deep Reinforcement Learning》为目标，选择
Breakout 做单任务、单 seed 的理解与部分复现。论文报告平均分 168、best 225，但现代 ALE、
启动随机化和 frame 计数存在不可消除的历史漂移，因此先检验训练趋势，不承诺逐分对齐。

本仓是复现 control repo，也跟踪独立实现和分析脚本。第三方代码、大型 run、checkpoint、ROM、
论文 PDF 和完整 artifact 均在 Git 外，通过 `research/repositories.yaml`、commit 和 hash 关联。

## 代码与环境

- 独立实现：`src/dqn2013_breakout.py`
- 源码 SHA256：`77aa269cd39888ebae4b0c256a120056141f96978284428801f2493e23ee62c0`
- Python env：`/root/autodl-tmp/envs/dqn2013`
- Atari：Gymnasium 0.29.1、ALE-Py 0.8.1、`BreakoutNoFrameskip-v4`
- GPU runtime：PyTorch 2.13.0+cu130，RTX 5090 capability 12.0 已实测
- 第三方参照：CleanRL commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`，MIT

## 2013 语义

- 两层卷积（16x8x8/stride4，32x4x4/stride2）+ 256 隐层；
- online Q bootstrap target，不使用延迟 target network；
- RMSProp、uniform replay、reward clipping、4-frame stack；
- epsilon 从 1.0 降到 0.1；独立 epsilon 0.05 evaluation；
- action repeat 4，不使用 Nature 版的 max-pooling 和 life-loss terminal。

随机 no-op 与 FIRE reset 是现代可运行性处理，optimizer 常数和 frame-budget 语义也存在历史
不确定性，均在 `references/CLAIM_PROTOCOL_MATRIX.md` 中登记。

## 当前结果

`EXP-0001` 已完成 10M emulator frames。epsilon=.05 评估峰值为 10.90，最终为 2.21；前者说明
确有学习，后者显示策略明显回退，两者都远低于论文平均 168。中文裁决与原始证据见
`/root/autodl-tmp/artifacts/dqn2013/EXP-0001/RESULT.md`。

`EXP-0003` 只修正 replay capacity 与 epsilon decay 的 agent-step 单位：终段 Q mean max 从
61.74 降到 3.72，配对终点评估从 2.51 提高到 10.04。该干预支持单位缩短是首次失稳的重要
促成因素，但单 seed、联合变量和 1.5M decisions 仍只支持 `promising_unresolved`，不构成论文
168 分的数值复现。

## 仓库角色

| 角色 | 位置 |
|---|---|
| control/runtime | 本仓与 `src/` |
| workflow | `/root/autodl-tmp/research-agent-kit@f3a120e`，tag `v0.2.0` |
| third-party | `/root/autodl-tmp/third_party/`，逐仓固定 commit |
| runs | `/root/autodl-tmp/runs/` |
| artifacts | `/root/autodl-tmp/artifacts/dqn2013/` |

本仓首次提交晚于三个已完成实验，历史边界见 `MIGRATION.md`。恢复状态时运行：

```bash
researchctl status
researchctl audit --strict
researchctl hygiene
```
