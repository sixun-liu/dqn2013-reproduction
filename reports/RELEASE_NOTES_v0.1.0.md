# v0.1.0 - Reproducible Nature 2015 DQN Breakout Release

This release packages the repository's single-seed Nature 2015 DQN Breakout
partial reproduction for independent verification.

## Headline result

- Paper reference: `316.8` (Nature 2015 Extended Data Table 3, replay + target)
- Local peak/final mean: `350.1833`
- Budget: 10M agent decisions / 40M nominal emulator frames
- Evaluation: 40 periodic evaluations; final checkpoint re-evaluated over 60 complete games
- Evidence level: single-task, single-seed, modern-ALE partial reproduction

## Release asset

`dqn2015-breakout-exp0004-s0-10m.pt`

SHA256:

```text
73e3e71f437bf07f59128b712f8a7e294c23052b9d6d5c62cb2478b58d672ef0
```

The checkpoint contains network, optimizer and RNG state. It does not contain
the replay buffer or ALE state and is therefore intended for evaluation, not
protocol-equivalent training resume.

Atari ROMs are not included. Follow `REPRODUCING.md` and explicitly accept the
separate AutoROM license before running tests, smoke, evaluation or training.

