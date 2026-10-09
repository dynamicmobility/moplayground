---
layout: default
title: Training
nav_order: 4
---

# Training

To train a pre-existing environment, check out the configuration files in `config/`. These files specify everything from model architecture and MORLAX parameters to reward and environment constants.

Choose the config file you want, edit the parameters to your liking, and run:

```bash
python3 -m scripts.train config_path
```

where `config_path` is the path to the config of your choice.

If you downloaded a policy in the past, you can also use those configs to run an identical training run on your system.

## Config layout

`src/moplayground/config.py` is the only code that reads these files. Scripts call `config.load(path)` and then getter functions such as `config.ppo_params(cfg)`; package functions take plain values only.

```yaml
name: test                   # run name; 'test' may overwrite its run folder and records no git hash
save_dir: results/...        # the run folder is save_dir/name
description: ''
algorithm: morlax            # morlax | amor
env: MOCheetah
backend: np                  # np | jnp (training always uses jnp)
env_config: {...}            # environment settings, given to the env class
learning_params:
  ppo_params:      {...}     # PPO settings, both algorithms
  sampling_params: {alpha, k, sampling, warmup_frac}   # preference sampling, both algorithms
  morlax_params:   {hypertype, hypersize, num_features, policy_hidden_layer_sizes, value_hidden_layer_sizes}
  amor_params:     {policy_hidden_layer_sizes, value_hidden_layer_sizes}
  morlax_warmup_params: {enabled, policy}               # not used yet
```

`config.load` also reads older layouts (for example `base_ppo_params`, `hypernetwork_params`, `algorithm: ppo`) and converts them, so saved `config.yaml` files from old runs still work. It checks the values and stops with a clear error if something is wrong.

Training writes a copy of the config to `save_dir/name/config.yaml`. Use that copy for rollouts, because it always matches the saved checkpoints.

## Writing your own training scripts

_Coming soon._
