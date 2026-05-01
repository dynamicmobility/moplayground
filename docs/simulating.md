---
layout: default
title: Simulating
nav_order: 3
---

# Simulating

## Downloading and evaluating policies

First, choose an environment from the following list:

- `cheetah`
- `hopper`
- `walker`
- `ant`
- `humanoid`
- `bruce`

Next, run the following script to download a policy corresponding to your selected environment:

```bash
python3 -m scripts.download_model --env cheetah
```

Note that you can supply a desired save directory via `--save_dir`, where the default directory is simply `results/wandb-downloads`.

Finally, you can run this policy and save a video of the rollout via:

```bash
python3 -m scripts.rollout_policy config_path
```

## Writing your own evaluation scripts

Evaluation (not used for training) scripts usually take the following form. Saving videos and plots has been omitted for conciseness.

```python
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path
from ral import BRUCE_TRADEOFFS


config = mm.utils.read_config()
env, env_params = mop.envs.create_environment(
    config,
    # add any env-specific kwargs here
)

camera = 'track'
directive = np.array([1.0, 0.0])

frames, reward_plotter, _, _ = mop.eval.rollout_policy(
    env         = env,
    config      = config,
    directive   = directive,
    T           = 6.0,
    camera      = camera,
    width       = 2560,
    height      = 1440
)
```
