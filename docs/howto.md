# Installation
First, clone the repository via SSH or HTTPS from [the moplayground repository](https://github.com/dynamicmobility/moplayground). 
Note that a `pip`-installable package is coming, and will be released after double-blind review!

Next, using the provided ymls, create a new conda environment.
**If you want to enable GPU based training, run:**

```bash
# Navigate to the moplayground directory
conda env create -f environment.yml
conda activate moplayground
```
**If you just want to evaluate policies and explore the code (i.e. running on a Mac):**

```bash
# Navigate to the moplayground directory
conda env create -f mac_environment.yml
conda activate moplayground
```
------
# Basic Usage
Several scripts have been provided that let you train and rollout policies.
Below, we'll show the primary usage of MO-Playground, evaluating and training policies.
## Downloading and evaluating policies
First, choose an environment from the following list:
- `cheetah`
- `hopper`
- `walker`
- `ant`
- `humanoid`
- `bruce`

Next, run the following script to download a policy corresponding to your selected environment
```python
python3 -m scripts.download_model --env cheetah  
```
note that you can supply a desired save directory via `--save_dir`, where the default directory is simply `results/wandb-downloads`.
Finally, you can run this policy and save a video of the rollout via 
```bash
python3 -m scripts.rollout config_path
```

## Training
To train a pre-existing environment, check out the configuration files in `config/`. 
These files specify everything from model architecture and MORLAX parameters to reward and environment constants.
Choose the config file you want, edit the parameters to your liking, and run 
```bash
python3 -m scripts.train config_path
```
where `config_path` is the path to the config of your choice.
If you downloaded a policy in the past, you can also use those configs to run an identical training run on your system.

-------
# Advanced Usage
For advanced usage of MO-Playground, it is generally assumed that you are familar with [MuJoCo Playground](https://playground.mujoco.org).
Specifically, how the `MjxEnv` class works, like the `step` and `reset` functions. 
It is also useful to take a look at how PPO works in [brax](https://github.com/google/brax) before diving into MORLAX.
## Writing your own evaluation scripts
Evaluation (not used for training) scripts usually take the following form.
Saving videos and plots has been omitted for conciseness.
```python
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import jax
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path
from ral import BRUCE_TRADEOFFS


from moplayground import config

cfg = config.load('path/to/config.yaml')
env, env_params = mop.create_environment(
    env_name     = config.env_name(cfg),
    env_params   = config.env_params(cfg),
    backend      = config.backend(cfg),
    gaitlib_path = config.gaitlib_path(cfg),
    # add any env-specific kwargs here
)

camera = 'track'
directive = np.array([1.0, 0.0])

inference_fn = mop.load_mo_policy(
    algorithm      = config.algorithm(cfg),
    network_params = config.network_params(cfg),
    num_objectives = config.num_objectives(cfg),
    run_dir        = config.run_dir(cfg),
    tradeoff       = directive,
    deterministic  = True
)
inference_fn = jax.jit(inference_fn)

frames, reward_plotter, _, _ = mm.eval.rollout_policy(
    inference_fn = inference_fn,
    env         = env,
    T           = 6.0,
    camera      = camera,
    width       = 2560,
    height      = 1440
)
```
## Writing your own training scripts
TODO
## Creating custom environments

Add a description and code example here.

```python
# example code here
```
