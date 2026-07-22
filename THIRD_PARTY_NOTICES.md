# Third-Party Notices

This repository contains an independent DQN implementation and does not
redistribute Atari ROMs, the DeepMind DQN Lua source, or third-party model
weights.

## CleanRL

The execution structure and Atari wrapper order in
`src/dqn2015_nature_breakout.py` were adapted from CleanRL's `dqn_atari.py` at
commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`.

CleanRL is Copyright (c) 2019 CleanRL developers and is distributed under the
MIT License. The complete notice is preserved in
`LICENSES/CleanRL-MIT.txt`.

## Visualizing and Understanding Atari Agents

The perturbation formula implemented independently in
`src/dqn2015_visual_intervention.py` follows the method and public MIT-licensed
reference implementation by Sam Greydanus at commit
`182492dab59da50aabe254e67e6d51c4f8622400`.

## DeepMind DQN 3.0

DeepMind DQN 3.0 at commit
`9d9b1d13a2b491d6ebd4d046740c511c662bbe0f` was consulted as a read-only
protocol oracle. Its limited academic-review source is not copied or
redistributed here.

## Atari ROMs

Atari ROMs are not included. `scripts/reproduce.sh setup` uses AutoROM only
after the caller explicitly accepts the ROM license with
`DQN_ACCEPT_ROM_LICENSE=1`.

