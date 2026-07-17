# DEVLOG

## 2026-07-17

- 固定目标为 2013 arXiv DQN，而非 2015 Nature DQN。
- 选择 Breakout，因为论文同时给出学习曲线和最终表格结果。
- CleanRL 仅作为第三方工程参照；运行代码采用独立 2013-style 实现。
- 现代 ALE 使用无 sticky 的 `BreakoutNoFrameskip-v4`，底层 frameskip 1，再由 wrapper repeat 4。
- PyTorch 2.13.0+cu130 在 RTX 5090 上完成 capability 12.0 矩阵 smoke；DQN 2K-step update smoke
  约 210 decisions/s，checkpoint/JSONL/completion 均有效。
- `EXP-0001` 已在无 outcome 条件下冻结；启动条件是 DreamerV3 释放 GPU 且 `nvidia-smi` 无 compute process。
- `EXP-0001` 前 4M emulator frames 的独立评估均值从 1.30 升至 8.47，下尾从 0 升到 4；
  online-target Q/loss 仍有限。观察支持早期学习趋势，但与论文平均 168 尚不接近。
- `EXP-0001` 自然完成 10M emulator frames，十次评估均值为
  1.30、2.37、6.75、8.47、7.08、2.51、2.49、3.20、10.90、2.21。策略学到过高于随机的行为，
  但回退明显且远低于论文平均 168；裁决为 `promising_unresolved`，明确不宣称数值复现。
- 最终 checkpoint 为 step 2.5M、reason `complete`；checkpoint 不含 replay，不能作为协议等价续跑。
- 冻结 manifest SHA256 `9c8b64c2...` 与 run 自动生成的 `config.json` SHA256 `3b75b538...`
  不同；逐项 diff 仅为前者额外携带 `source` 和 `source_sha256` 两个 provenance 字段，所有可执行
  超参数一致，不构成运行配置漂移。
- Post-run 协议审计发现 EXP-0001 只产生约 2.45M updates，而 primary Figure 2 覆盖约 5M；
  replay/epsilon 也把论文 1M `frames` 除以 repeat 得到 250K，与 2015 作者代码和 NIPS 第三方
  对 agent-step 的解释冲突。因此本地学习/退化观察有效，但与论文 168 的直接数值比较 provenance 不足。
- `EXP-0002` 用最终 checkpoint 在 10 个冻结 seed 上评估：seed mean 2.21--2.46，中位 2.26，
  共 472 episodes；确认末点策略普遍退化，排除单一评估 seed 偶然。
- 新 instrumented executor 与旧 executor 做 2K-step parity：模型和优化器逐 tensor 完全相等。
  `EXP-0003` 只把 replay/epsilon 从 250K 修正到 1M agent steps；100K pilot 增量吞吐 227.6/s，
  ETA 约 1.7h，继续到 1.5M decisions。
- `EXP-0003` 于 11:51 CST 自然完成 1.5M decisions。预注册终段 51 个 logged samples 中，
  q_mean max 3.72、loss max .242；EXP-0001 同窗口为 61.74 和 25.40。终点配对 legacy mean
  从 2.51 提高到 10.04，固定 seed20000 mean 8.23，支持联合单位修正避免首次退化窗口。
- 裁决为 `promising_unresolved`：单 seed、replay/epsilon 联合轴和 1.5M 短预算不能升级为论文
  数值复现。暂不碰 optimizer/gamma 或 2015 target network；长程验证需从头冻结，因为 checkpoint
  不保存 replay，不能协议等价续跑。
