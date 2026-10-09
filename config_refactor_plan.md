# Config refactor plan

## Goal

- Only one file knows YAML key names: `src/moplayground/config.py`.
- The `moplayground` package never receives the full config. Package functions take plain values.
- Scripts in `scripts/` load the YAML, read values with getter functions, and pass the values to the package.

## Canonical YAML layout

```yaml
name: ...
save_dir: ...
description: ...
algorithm: morlax            # morlax | amor
env: MOCheetah
backend: np                  # np | jnp (training always uses jnp)
gaitlib_path: ...            # NaviGait only
env_config: {...}            # unchanged; given to the env class as env_params
learning_params:
  ppo_params:      {learning_rate, num_envs, num_timesteps, ...}   # both algorithms
  sampling_params: {alpha, k, sampling, warmup_frac}               # both algorithms
  morlax_params:   {hypertype, hypersize, num_features,
                    policy_hidden_layer_sizes, value_hidden_layer_sizes}
  amor_params:     {policy_hidden_layer_sizes, value_hidden_layer_sizes}
  morlax_warmup_params: {enabled, policy}   # unused; see TODO in config.py
```

Policy and value layer sizes stay separate for MORLAX and AMOR. They size different networks:
MORLAX sizes the network that the hypernetwork generates; AMOR sizes a network that trains directly.

## New file: `src/moplayground/config.py`

### `load(path)`

1. Read the YAML file into a ConfigDict. This replaces `minimal_mjx.utils.read_config`.
2. Rename old keys to the canonical keys, so saved configs from old runs still load.
   Known old names:
   - `base_ppo_params` -> `ppo_params`
   - `hypermorl_params` -> `sampling_params`
   - `hypernetwork_params` + `network_params` (layer sizes) -> `morlax_params`
   - `algorithm: ppo` or missing -> `morlax`
   - `hypertype: ActorCritic` -> `dual` (renamed in commit 899b126)
   - sampling: when both algorithms have a block, keep the block of the configured algorithm
   - top-level `network_params` (MORLAX shape) -> `morlax_params`
   - `morlax_params.{alpha,k,sampling,warmup_frac}` -> `sampling_params`
   - `amor_params.train_fn_params` -> `sampling_params`
   - `amor_params.network_params` -> `amor_params`
   - `warmup_params` -> `morlax_warmup_params`
3. Check values and raise a clear error on failure:
   - `algorithm` is `morlax` or `amor`.
   - `env` is a known environment.
   - The required `learning_params` sections exist for the chosen algorithm.
   - `sampling_params.sampling` is a known sampling mode.
   - Every key in `env_config.reward.optimization.objectives` and `shared_objectives` is in `reward.weights`.
   - `len(labels) == len(objectives)`.

### Getters

| Function | Returns |
|---|---|
| `algorithm(cfg)` | `'morlax'` or `'amor'` |
| `env_name(cfg)` | env class name |
| `backend(cfg, for_training)` | `'jnp'` if training, else `cfg.backend` |
| `env_params(cfg)` | `env_config` as a ConfigDict |
| `gaitlib_path(cfg)` | NaviGait gait library path |
| `ppo_params(cfg)` | dict |
| `sampling_params(cfg)` | dict |
| `network_params(cfg)` | `morlax_params` or `amor_params`, chosen by `algorithm` |
| `num_objectives(cfg)` | int |
| `objective_labels(cfg)` | list of str |
| `run_dir(cfg)` | `Path(save_dir) / name` |

### Setters (for `scripts/ablation.py`)

- `set_hypertype(cfg, value)`
- `set_sampling(cfg, value)`
- `set_k(cfg, value)`
- `set_name(cfg, value)`

### TODO comment in `config.py`

Add this comment where `config.py` handles `morlax_warmup_params`:

```python
# TODO: nothing reads morlax_warmup_params yet; implement policy warmup or remove it.
```

### `save(cfg, run_dir)`

Write `config.yaml` and the git hash into the run folder. Moves out of `learning/training.py`.

## Package changes (plain values only)

| Function | Today | After |
|---|---|---|
| `envs.create_environment` | `config` | `env_name, env_params, backend, gaitlib_path=None, **env_kwargs` |
| `learning.train_policy` | `config` | `algorithm, ppo_params, sampling_params, network_params, run_dir, env, eval_env, ...` |
| `learning.inference.load_mo_policy` | `config` | `algorithm, network_params, num_objectives, run_dir, tradeoff, ...` |
| `learning.inference.get_num_objectives` | `config` | remove (use `config.num_objectives`) |
| `eval.pareto.*` | `config` / list of configs | plain values per run (`algorithm, network_params, num_objectives, run_dir`) |
| `eval.simulate` | calls `read_config()` itself | skipped: `# TODO: talk to neil` (imports already broken) |

Checkpoint lookup (`mm.learning.inference.get_last_model`) takes a config today.
The package needs a local version that takes `run_dir`.

Env classes still receive `env_params` (the `env_config` section). Out of scope for this refactor.

## Script changes

Each script in `scripts/` does:

```python
cfg = config.load(path)
value = config.<getter>(cfg)
package_function(value, ...)
```

- `train.py`, `rollout.py`, `draw_frontier.py`, `ablation.py`.
- `ablation.py` uses the setters.
- `rollout.py` and `draw_frontier.py` take the YAML path as an explicit CLI argument.

## wandb and saved config

Today, the config reaches wandb in two ways:

1. `scripts/train.py` and `scripts/ablation.py` call `wandb.init(config=...)` with the full config.
   wandb shows these values as the run's hyperparameter table.
2. `train_policy` (`learning/training.py`) writes `config.yaml` (plus git hash) into `save_dir/name`,
   then calls `run.log_artifact(config.yaml)` to upload the file.

After the refactor, step 2 moves out of the package into the script. `train_policy` gets only
`run` and `run_dir`. It still uses `run` to log progress and checkpoints, as today.

```python
# scripts/train.py
cfg     = config.load(args.config)
run_dir = config.run_dir(cfg)
config.save(cfg, run_dir)                 # makes run_dir, adds git hash, writes config.yaml

run = mm.utils.logging.initialize_wandb(
    name    = str(run_dir).replace('/', ''),
    entity  = 'njanwani-gatech',
    project = 'PrefMORL',
    config  = cfg.to_dict(),              # canonical layout, includes git_hash
)
run.log_artifact(str(run_dir / 'config.yaml'), name='config')

mop.train_policy(
    algorithm       = config.algorithm(cfg),
    ppo_params      = config.ppo_params(cfg),
    sampling_params = config.sampling_params(cfg),
    network_params  = config.network_params(cfg),
    objective_labels= config.objective_labels(cfg),
    run_dir         = run_dir,
    run             = run,
    env=env, eval_env=eval_env,
)
```

Notes:
- `config.save` runs before `wandb.init`, so the wandb table now also gets `git_hash`. Today it does not.
- wandb and `config.yaml` both get the canonical layout, because `load()` renames old keys first.
- The `name == 'test'` rule moves from `create_training_directory` into `config.save`, unchanged. Both parts stay:
  1. `name != 'test'` and `run_dir` exists: stop with an error. `name == 'test'`: reuse and overwrite `run_dir`.
  2. `name == 'test'`: do not record `git_hash`.
- `ablation.py` repeats the same lines once per combo, after the setters.

## YAML changes

Convert every file in `config/` and `config/amor/` to the canonical layout.

## Out of scope / known breakage

- `ral/` scripts are not updated. They break where they call changed package functions.
- Env internals keep reading `self.params.*`.
- Update `CLAUDE.md` and `docs/` to the new layout; regenerate `docs/api/`.

## Verification

Main idea: record what the package receives **before** the refactor, then check that it receives the same values **after**.

1. **Baseline (before any change).** A scratch script runs the current code for every YAML in `config/` and `config/amor/`.
   It saves to JSON: the merged train-fn kwargs, the network-factory kwargs, `normalize_observations`,
   objective labels, `num_objectives`, env name, backend and `run_dir`.
2. **Equivalence (after the change).** The same script runs the new path (`config.load` + getters).
   The JSON output must match the baseline, except for renamed keys.
3. **Old saved configs.** `load()` must read every saved `config.yaml` (currently 3:
   `accepted-results/jun/17/test`, `results/wandb-downloads/cheetah`, `maako_optimization/test`).
4. **Same policy output.** Load the checkpoint in `accepted-results/jun/17/test` with the old code and with the new code.
   Run both policies on the same observations and the same random key. The actions must be identical. CPU is enough.
5. **Validation errors.** Bad configs must fail in `load()` with a clear message:
   `algorithm: so_ppo`, unknown env, unknown sampling mode, objective key missing from `reward.weights`, wrong label count.
6. **No config reads left in the package.** grep `src/moplayground/` (except `config.py`) for `config[`, `learning_params`, `read_config`, `save_dir`.
7. **Smoke training** (needs a GPU node; this login node has no GPU). Very small `num_timesteps`, `WANDB_MODE=offline`, so nothing uploads.
   Check: training finishes; `config.yaml` has the canonical layout; `name: test` runs twice without error and has no `git_hash`;
   another name stops with an error on the second run and has `git_hash`; the offline wandb run has the config table and the artifact.
8. **Ablation.** Call `apply_overrides` and check the 3 values. Then one very small combo, offline.
9. **Rollout.** `scripts/rollout.py` on the old saved run and on the new smoke run.

Not tested: `ral/` (out of scope, known to break).

## Order of work

1. ✅ Write `config.py` (`load`, rename step, checks, getters, setters, `save`).
2. ✅ Change package function signatures to plain values.
3. Update `scripts/`.
4. ✅ Convert YAML files.
5. ✅ Update `CLAUDE.md`, `docs/`, regenerate `docs/api/`.
6. Smoke test: train for a few steps, roll out an old saved run and a new run.
