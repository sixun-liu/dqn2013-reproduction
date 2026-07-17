# TODO

- [x] 运行 2K-step update smoke 并测初始 ETA。
- [x] 按 12 小时总预算冻结正式终点和停止条件。
- [x] DreamerV3 释放 GPU 后启动 `EXP-0001`。
- [x] 在 100K decisions 用含 update 的稳态 SPS 再校准 ETA。
- [x] 运行并保存模型 checkpoint、JSONL 和 stdout。
- [x] 生成论文对照图和裁决。
- [x] 完成 2013 primary / 2015 author code / NIPS third-party / 本地实现协议审计。
- [x] 用 `EXP-0002` 十个固定 seed 排除最终低分的单一评估 seed 偶然。
- [x] 完成 `EXP-0003` replay/epsilon agent-step 单位修正诊断并与历史窗口比较。
- [ ] 用户人工复核 `artifacts/dqn2013/review/LATEST.md`。
- [x] 暂不测试 NIPS optimizer/gamma；target network 保持 parked。
- [ ] 经用户确认后，预注册协议对齐的约 5M-update 长程基线，或在当前部分复现处停止。
