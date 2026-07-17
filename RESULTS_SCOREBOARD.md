# RESULTS_SCOREBOARD

> Updated: 2026-07-17T05:26:23Z
> Maintainer: codex
> Source of truth: research/experiments.jsonl and research/artifacts.jsonl

| Run | Budget | Train return | Eval return | Verdict |
|---|---:|---:|---:|---|
| `EXP-0001` / seed 0 | 10M emulator frames | final rolling-20 7.9；p99.5 23 | peak 10.90 @ 9M；final 2.21；paper average 168 | `promising_unresolved`；数值未复现 |
| `EXP-0002` / fixed-seed reevaluation | final checkpoint；10 frozen seeds | N/A | seed mean 2.21--2.46；472 episodes | 排除单一评估 seed 偶然；末点退化成立 |
| `EXP-0003` / seed 0 | 1.5M agent decisions | final-window Q mean max 3.72；loss max 0.242 | legacy 10.04；fixed seed 8.23 | `promising_unresolved`；联合单位修正避免首次失稳窗口 |
