# 首次 Git 化说明

迁移时间：2026-07-17 UTC。

本仓首次 Git 提交晚于 `EXP-0001`、`EXP-0002` 和 `EXP-0003` 的运行与结案。首次提交是既有控制
材料、独立实现和分析脚本的 post-run import，不是任何历史实验运行前已经存在的 commit。

## 历史源码锚点

| 文件 | SHA256 | 用途 |
|---|---|---|
| `src/dqn2013_breakout.py` | `77aa269cd39888ebae4b0c256a120056141f96978284428801f2493e23ee62c0` | `EXP-0001` 独立实现 |
| `src/dqn2013_breakout_instrumented.py` | `b8bea12b77434f59e69a80f974269b56d5ee88fa89d00dc930c81f6578e16f6a` | `EXP-0003` instrumented executor |
| `scripts/analyze_dqn2013_unitfix.py` | `229d639367fca76a7869b5f460b61e02404f7bb1e832d56fc39e5867d76ea454` | `EXP-0003` 结案分析 |

历史 freeze、artifact registry 和 run 内配置继续提供各实验的一手 provenance。不得用首次 Git
提交反向替代这些 hash，也不得声称它是 pre-run commit。已有配置名中的日期后缀属于历史路径，
为避免破坏 registry 不做重命名；新实验开始使用 `EXP-ID` 前缀和 UTC run tag。

从下一实验开始，formal/replication 必须在 clean commit 上冻结，并分别记录 control、runtime、
workflow 和 third-party provenance。

