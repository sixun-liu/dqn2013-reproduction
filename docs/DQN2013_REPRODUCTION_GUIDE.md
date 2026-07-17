# 2013 DQN 理解与复现导读

## 1. 论文的关键贡献

2013 DQN 把卷积网络、Q-learning 和 experience replay 接到 Atari 原始像素上。最重要的不是
单独某个网络结构，而是三件事组合：从像素学习状态表示、用 replay 打散相邻样本相关性、用
off-policy Q-learning 重复利用历史数据。论文在七个游戏上共用网络与超参数，Breakout 报告
平均分 168、best 225，随机策略为 1.2。

## 2. 一次更新发生什么

```text
最近 4 帧 -> 卷积网络 -> 每个动作的 Q 值
epsilon-greedy 选择动作 -> 连续执行 4 个 emulator frames
转移写入 uniform replay
从 replay 随机采样 32 条转移
target = reward + gamma * max_a Q(next_state, a)
最小化 target 与 Q(state, action) 的均方误差
```

论文网络为 Conv 16@8x8/stride4、Conv 32@4x4/stride2、FC 256，再输出每个合法动作的 Q 值。
训练 reward 裁剪到 -1/0/1，但评估报告原始游戏分数。

## 3. 2013 与 2015 Nature DQN 不能混用

| 项目 | 2013 arXiv | 2015 Nature / CleanRL 参照 |
|---|---|---|
| Target | 当前 online Q（对 target stop-gradient） | 延迟更新的 target network |
| 网络 | 两层卷积 + FC 256 | 三层卷积 + FC 512 |
| Atari 游戏 | 7 | 49/57 口径 |
| 训练预算 | 论文写 10M frames，历史计数语义有歧义 | 常见 200M emulator frames |
| CleanRL 默认 | 不适用 | Adam、target network、Nature-style wrappers |

CleanRL commit `fe8d8a0` 的文档也明确以 2015 论文为目标。本次只参考它的单文件工程结构；实际
运行 `src/dqn2013_breakout.py` 是独立 2013-style 重实现。

## 4. 本次协议

- `BreakoutNoFrameskip-v4`：sticky action probability 0，最小动作集 4，底层 frameskip 1；
- wrapper 负责 repeat 4，不做 Nature 版 max-pooling，不把 loss of life 当 terminal；
- grayscale -> 110x84 -> crop 84x84，堆叠最近 4 帧；
- online bootstrap target、RMSProp、uniform replay、reward clipping；
- epsilon 在 250K decisions 内从 1.0 降到 0.1；
- 每 250K decisions 单独用 epsilon 0.05 跑 10K decisions；
- 最多 2.5M decisions = 本项目保守定义的 10M emulator frames，seed 0。

## 5. 不可消除的历史不确定性

- 论文说“10M frames”，但老 ALE 文献有时把 agent decision、预处理帧和原始 emulator frame
  混用；本项目选择最保守的原始帧换算，所以可能短于原实现的实际训练量。
- 论文只写 RMSProp，没有完整列出所有 optimizer 常数；本地采用历史 DQN 常见值并标成实现选择。
- random no-op 与 FIRE reset 是现代可运行性处理，论文正文没有完整冻结这些 wrapper 细节。
- ALE、Gymnasium、PyTorch 和 GPU 均为现代版本。因此这不是 exact artifact replication。

## 6. 结果能说明什么

独立评估回报持续上升并明显超过 1.2，可支持“2013 算法机制在现代环境中产生论文所述学习
趋势”。进入 168--225 是更强证据，但单 seed、现代环境和预算歧义仍不足以宣称数值复现。
若 online target 出现 Q 值爆炸或性能崩溃，这也是有效观察；应先检查协议与数值稳定性，不能
事后悄悄加入 2015 target network 再称为 2013 结果。

