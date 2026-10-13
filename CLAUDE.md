# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`moplayground` is a Python project for **massively parallelized multi-objective reinforcement learning for robotics** by Neil Janwani, Ellen Novoseller, Vernon Lawhern, and Maegan Tucker.

Top-level layout:

- `src/moplayground/` — the installable `moplayground` package (training envs, algorithms, utilities). Installed via `pyproject.toml`.
- `ral/` — analysis / plotting / video scripts (Pareto fronts, t-SNE, hypervolume, composites, etc.). Run as standalone scripts, not a package.
- `scripts/` — entry points: `train.py`, `train_ppo.py` (older standalone PPO baseline), `rollout.py`, `draw_frontier.py`, `download_model.py`, and `build_api_docs.py` (docs generator). `train.sh` wraps `train.py`.
- `config/` — training/experiment configs.
- `externals/` — vendored or third-party deps.
- `environment.yml` / `mac_environment.yml` — conda envs (Linux/CUDA vs. macOS).
- `pyproject.toml` — Python package metadata for `moplayground`.
- `output/`, `results/`, `wandb/`, `dist/` — generated artifacts; do not hand-edit.
- `docs/` — the project's static website (academic landing page + Jekyll docs site). See `docs/` for its own conventions.

## Common commands

Set up the environment (first time):
```bash
conda env create -f environment.yml          # Linux / CUDA
conda env create -f mac_environment.yml      # macOS
pip install -e .                             # install moplayground in editable mode
```

Train / roll out:
```bash
python -m scripts.train config/mocheetah.yaml               # or: bash scripts/train.sh config/mocheetah.yaml
python -m scripts.rollout <save_dir>/<name>/config.yaml     # roll out a trained run (use the run's saved config)
python -m scripts.draw_frontier <save_dir>/<name>/config.yaml  # evaluate a trained run's Pareto front
```

## Config files

`src/moplayground/utils/config.py` defines `MOConfig` (a subclass of `minimal_mjx.Config`), the **only** place that knows YAML key names. Scripts build `cfg = mop.MOConfig(path)` and read properties (`cfg.algorithm`, `cfg.ppo_params`, `cfg.sampling_params`, `cfg.morlax_network_params`, `cfg.amor_network_params`, `cfg.ppo_network_params`, `cfg.default_scalarization`, `cfg.run_dir`, ...), then pass plain values to package functions. Package functions never receive a config. `cfg.save(cfg.run_dir / 'config.yaml')` writes the run's config (adds a git hash unless the run is named `test`). Canonical layout:

```yaml
env_config:
  reward:
    optimization:
      objectives:            [[...], ...]       # reward keys per objective
      shared_objectives:     [...]              # added to every objective
      labels:                [...]              # display names
      default_scalarization: [1, 1, 1]          # weights used when algorithm: ppo
learning_params:
  ppo_params:            {...}                                   # all algorithms
  sampling_params:       {alpha, k, sampling, warmup_frac}       # morlax / amor
  morlax_network_params: {hypertype, hypersize, num_features, policy_hidden_layer_sizes, value_hidden_layer_sizes}
  amor_network_params:   {policy_hidden_layer_sizes, value_hidden_layer_sizes}
  ppo_network_params:    {policy_hidden_layer_sizes, value_hidden_layer_sizes}
  morlax_warmup_params:  {enabled, policy}                       # unused (TODO)
```

When adding a config key, add a property on `MOConfig` and never index the config elsewhere.

## MORL algorithms

`moplayground` ships two multi-objective RL algorithms, both PPO-based, both using directive (tradeoff) scalarization of per-objective rewards. Selected via `algorithm:` in the YAML config (`morlax`, `amor`, or `ppo`).

- **MORLAX** (`src/moplayground/moppo/morlax.py`) — *hypernetwork* approach. A hypernet maps directive → policy/value MLP weights. The base policy/value MLPs are not trained directly; only the hypernet is. Network shape (`hypertype`, `hypersize`, `num_features`, plus target `policy_hidden_layer_sizes` / `value_hidden_layer_sizes`) is configured under `learning_params.morlax_network_params`. `hypertype` is `single` (shared feature MLP, separate W/b heads → `ActorCriticHypernet`) or `dual` (separate feature MLPs per head → `DualA2CHypernet`).
- **AMOR** (`src/moplayground/moppo/amor.py`) — *tradeoff-conditioned policy* baseline. The directive is concatenated to the (normalized) obs and fed into flat policy/value MLPs. No hypernetwork. Network shape is configured under `learning_params.amor_network_params` (`policy_hidden_layer_sizes`, `value_hidden_layer_sizes`).

- **PPO** (`algorithm: ppo`) — single-objective baseline. `scripts/train.py` wraps the envs in `Multi2SingleObjective` (`envs/generic/mobase.py`) to dot the reward vector with `default_scalarization`, then trains with `mm.setup_ppo`. Network shape is under `learning_params.ppo_network_params`.

MORLAX and AMOR share `learning_params.ppo_params` (PPO settings) and `learning_params.sampling_params` (`alpha`, `k`, `sampling`, `warmup_frac`).

Shared infrastructure lives in `src/moplayground/moppo/`:
- `factory.py` — `make_morlax_networks`, `make_amor_networks`, and their inference-fn factories (`make_hypernetwork_inference_fn`, `make_amor_inference_fn`).
- `losses.py` — `compute_morlax_loss`, `compute_amor_loss`, plus `MORLAXNetworkParams` / `AMORNetworkParams`.
- `acting.py` — shared `MultiObjectiveTransition`, `actor_step`, `generate_unroll`, `Evaluator`.
- `networks.py` — hypernet variants (`Hypernet`, `HypernetMLP`, `DualA2CHypernet`, etc.).

Training goes through `mm.train` (minimal_mjx). `scripts/train.py::get_handle_params` picks the setup function (`mm.setup_ppo`, `mop.setup_morlax`, `mop.setup_amor` from `learning/training.py`) plus its train-fn and network-factory params, and `get_progress` picks the progress callback class (`mm.Progress` for PPO, `mop.MOProgress` from `utils/plotting.py` otherwise). Setup functions have the signature `setup_x(train_fn_params, network_factory_params, resumed) -> (train_fn, network_factory)`; the MO ones pass `wrap_env_fn=mo_wrapper`. Inference dispatch is in `learning/inference.py::load_mo_policy` (`morlax` / `amor` only). AMOR's inference fn natively takes the directive at call time (`policy(obs, directive, key)`); `load_mo_policy` returns a 2-arg `policy(obs, key)` with the tradeoff baked in for compatibility with `mm.eval.rollout_policy`. To switch the directive at runtime, use `make_amor_inference_fn` directly.

Checkpoint formats differ:
- MORLAX: `(normalizer, hypernet)` 2-tuple.
- AMOR: `(normalizer, policy, value)` 3-tuple — matches the standard brax layout.

Regenerate API docs (writes Markdown into `docs/api/` from `moplayground` docstrings via `lazydocs`):
```bash
python scripts/build_api_docs.py
```

Preview the docs site:
```bash
cd docs
bundle install                               # first time only
bundle exec jekyll serve                     # add --port 4001 if 4000 is taken
```

## Editing notes

- `docs/api/` is **auto-generated** — do not hand-edit. Edit docstrings in `src/moplayground/`, rerun `scripts/build_api_docs.py`, and commit the regenerated Markdown.
- `ral/` scripts are run-as-script utilities; expect `if __name__ == "__main__":` entry points and CLI args, not importable APIs.
- The docs site uses the `just-the-docs` Jekyll remote theme; the academic landing page (`docs/index.html`) is a Bulma-based template that bypasses the Jekyll layout (`layout: null`, `nav_exclude: true`).
- When commands or APIs documented in `docs/` drift from the actual code in `src/moplayground/` or `scripts/`, the code is authoritative — update the docs to match.
