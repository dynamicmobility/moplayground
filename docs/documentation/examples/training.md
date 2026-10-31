# Training a policy

`scripts/train.py` trains a policy for any registered environment with any of the
three algorithms (`morlax`, `amor`, `ppo`). Everything about the run, including the
algorithm, is read from a single YAML config.

```bash
python -m scripts.train config/mocheetah.yaml          # MORLAX on the Cheetah
python -m scripts.train config/amor/mocheetah.yaml     # AMOR on the Cheetah
```

The script takes one positional argument, the path to the config. To train with
plain PPO, set `algorithm: ppo` in the config; the objectives are then combined with
the fixed weights in `default_scalarization`.

## Step by step

### 1. Read the config

```python
cfg = mop.MOConfig(config_path)
```

[`MOConfig`][moplayground.utils.config.MOConfig] extends `minimal_mjx.Config` with the
multi-objective sections (`sampling_params`, `morlax_network_params`, `objectives`,
...). It is the only code that knows the YAML layout; every library function takes
plain values. See [The config file](../guides/configuration.md).

### 2. Build the training and evaluation environments

```python
env_kwargs = dict(
    env_name     = cfg.env_name,
    backend      = 'jnp',
    gaitlib_path = cfg.gaitlib_path,
)
env, env_cfg = mop.create_environment(env_params=cfg.env_params, **env_kwargs, **TRAIN_KWARGS)
eval_env, _  = mop.create_environment(env_params=cfg.env_params, **env_kwargs, **EVAL_KWARGS)
if cfg.algorithm == 'ppo':
    env         = mop.Multi2SingleObjective(env, cfg.default_scalarization)
    eval_env    = mop.Multi2SingleObjective(eval_env, cfg.default_scalarization)
```

[`create_environment`][moplayground.envs.create.create_environment] maps the config's
`env` name to an environment class. Training always uses the `jnp` (JAX/MJX) backend,
whatever the config's `backend` says. `TRAIN_KWARGS` and `EVAL_KWARGS` carry
environment-specific constructor arguments; for Bruce, evaluation uses a fixed
command speed.

For `ppo`, [`Multi2SingleObjective`][moplayground.envs.generic.mobase.Multi2SingleObjective]
replaces the vector reward with `reward · default_scalarization`, so the environment
can be trained with standard single-objective PPO.

### 3. Save the config and start Weights & Biases

```python
cfg.save(cfg.run_dir / 'config.yaml', warn_github_changes=False)

run = mm.initialize_wandb(
    name    = str(cfg.run_dir).replace('/', ''),
    entity  = 'njanwani-gatech',
    project = 'MO-Playground-2',
    config  = cfg.cfg.to_dict()
)
```

The run directory is `save_dir/name`. The saved `config.yaml` is the one to pass to
`rollout.py` later; unless the run is named `test`, it also records the current git
commit as `git_hash`.

!!! note "Weights & Biases"
    The W&B entity and project are set in `train.py`; change them to your own before
    training. To train without logging, set `WANDB_MODE=disabled` in the environment.

### 4. Select the algorithm

```python
def get_handle_params(cfg: mop.MOConfig):
    match cfg.algorithm:
        case 'ppo':
            return mm.setup_ppo, cfg.ppo_params, cfg.ppo_network_params
        case 'morlax':
            return mop.setup_morlax, cfg.ppo_params | cfg.sampling_params, cfg.morlax_network_params
        case 'amor':
            return mop.setup_amor, cfg.ppo_params | cfg.sampling_params, cfg.amor_network_params

def get_progress(cfg: mop.MOConfig):
    match cfg.algorithm:
        case 'ppo':
            return mm.Progress
        case _:
            return mop.MOProgress
```

Each algorithm is described by three values:

- a **setup function** `setup_x(train_fn_params, network_factory_params, resumed)`
  that returns `(train_fn, network_factory)`. [`setup_morlax`][moplayground.learning.training.setup_morlax]
  and [`setup_amor`][moplayground.learning.training.setup_amor] bind the parameters
  into [`morlax.train`][moplayground.moppo.morlax.train] or
  [`amor.train`][moplayground.moppo.amor.train], and set the environment wrapper to
  [`mo_wrapper`][moplayground.learning.training.mo_wrapper];
- the **train-function parameters**: the PPO settings, plus the tradeoff sampling
  settings for the multi-objective algorithms;
- the **network-factory parameters** for that algorithm's networks.

The progress callback is [`MOProgress`][moplayground.utils.plotting.MOProgress] for the
multi-objective algorithms. After each evaluation it records the evaluation Pareto
front, appends its hypervolume and sparsity to `progress.csv`, logs both to W&B, and,
for two objectives, plots the sequence of fronts to `progress.svg`.

### 5. Train

```python
mm.train(
    env                     = env,
    eval_env                = eval_env,
    run_dir                 = cfg.run_dir,
    handle_params           = handle_params,
    train_fn_params         = train_fn_params,
    network_factory_params  = network_factory_params,
    progress_cls            = progress,
    run                     = run,
    resume                  = False,
    overwrite               = cfg.name == 'test'
)
```

`mm.train` (from minimal_mjx) creates the run directory, calls the setup function,
attaches the progress and checkpoint callbacks, and runs the trainer. A run directory
that already holds training output is not overwritten unless the run is named `test`.

During MORLAX and AMOR training, every PPO iteration samples a batch of tradeoffs
with [`sample_preferences`][moplayground.moppo.morlax.sample_preferences] and assigns
one to each parallel environment. The `sampling_params` control this:

| Key | Meaning |
| --- | --- |
| `sampling` | `dense` (one Dirichlet sample per environment), `sparse` (`k` samples, each shared by `num_envs / k` environments), `sparse-heavytail` (as `sparse`, with the one-hot corners of the simplex always included), or `single-avg` (uniform weights only) |
| `alpha` | Dirichlet concentration. Values above 1 favour balanced tradeoffs; below 1, extreme ones. |
| `k` | Number of distinct tradeoffs per iteration for the `sparse` modes. `num_envs` must be divisible by it. |
| `warmup_frac` | Fraction of the evaluation epochs trained with uniform weights before sampling begins |

## Running in tmux

`scripts/train.sh` opens a detached `tmux` session named `MOPLAYGROUND`, activates the
`moplayground` conda environment, logs in to W&B, and runs `train.py`:

```bash
bash scripts/train.sh config/mocheetah.yaml
```

## What a finished run looks like

```text
<save_dir>/<name>/
├── config.yaml          # the config this run used (pass it to rollout.py)
├── progress.csv         # one row per evaluation: time, step, hypervolume, sparsity
├── progress.svg         # sequence of evaluation fronts (two objectives only)
├── wandb_run_id.txt     # the W&B run this directory belongs to
├── 000000000000/        # one checkpoint per evaluation, named by environment step
├── 000003573760/
├── ...
└── 000050032640/        # newest checkpoint; rollout.py loads this one
    ├── ppo_network_config.json
    └── ...              # Orbax checkpoint files
```

Checkpoints are written once per evaluation (`num_evals` in total). A MORLAX checkpoint
stores `(normalizer, hypernetwork)` parameters; an AMOR checkpoint stores
`(normalizer, policy, value)`, the standard Brax layout.

## Full script

??? example "scripts/train.py"

    ```python
    --8<-- "scripts/train.py"
    ```
