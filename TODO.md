# TODO

> Updated: 2026-07-22T03:56:00Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期未完成动作；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 预注册局部 calibration probe，绑定9.0M/9.25M/9.5M/10M与训练 wrapper 语义；
  trigger: EXP-0005 closed and source clean。
- [ ] [codex] 先跑单阶段最小轨迹验证 reward/terminal/discount 与信号完整性，再执行四阶段；
  trigger: calibration probe frozen。
- [ ] [user] 复核 EXP-0005 固定状态图谱；trigger: 打开 `ART-0028`，不阻塞下一 diagnostic probe。
- [ ] [user] 复核 EXP-0004 既有主图；trigger: 打开 `ART-0019`，不阻塞 diagnostic probe。

## Waiting

- [ ] [codex] 四帧消融与带对照 saliency；trigger: calibration 确认9.25M局部异常。
- [ ] [codex] Target Network 因果消融；trigger: offline calibration/视觉证据闭环并形成单变量假说。
