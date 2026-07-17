# PLAN

1. 冻结 2013 DQN Breakout 协议和源码 hash。
2. 运行 smoke/pilot，验证 replay、训练、评估、checkpoint 和数值稳定性。
3. 在 DreamerV3 运行结束后的剩余墙钟预算内训练至最多 2.5M agent decisions。
4. 生成训练/评估曲线，与论文随机分 1.2、平均分 168、best 225 分层比较。
5. 按 provenance 和预算完整性裁决，不用现代 CleanRL 曲线替代论文证据。

