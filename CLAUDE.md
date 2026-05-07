# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`moplayground` is a Python project for **massively parallelized multi-objective reinforcement learning for robotics** by Neil Janwani, Ellen Novoseller, Vernon Lawhern, and Maegan Tucker.

Top-level layout:

- `src/moplayground/` — the installable `moplayground` package (training envs, algorithms, utilities). Installed via `pyproject.toml`.
- `ral/` — analysis / plotting / video scripts (Pareto fronts, t-SNE, hypervolume, composites, etc.). Run as standalone scripts, not a package.
- `scripts/` — entry points: `train.py`, `rollout.py`, `draw_frontier.py`, `download_model.py`, and `build_api_docs.py` (docs generator). `train.sh` wraps `train.py`.
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
python -m scripts.train config/mocheetah.yaml   # pass YAML config path directly
python -m scripts.rollout                       # or: bash scripts/train.sh
```

## MORL algorithms

`moplayground` ships two multi-objective RL algorithms, both PPO-based, both using directive (tradeoff) scalarization of per-objective rewards. Selected via `algorithm:` in the YAML config.

- **MORLAX** (`src/moplayground/moppo/morlax.py`) — *hypernetwork* approach. A hypernet maps directive → policy/value MLP weights. The base policy/value MLPs are not trained directly; only the hypernet is. Configured under `learning_params.morlax_params` (`hypertype`, `hypersize`, `num_features`, plus target `policy_hidden_layer_sizes` / `value_hidden_layer_sizes`).
- **AMOR** (`src/moplayground/moppo/amor.py`) — *tradeoff-conditioned policy* baseline. The directive is concatenated to the (normalized) obs and fed into flat policy/value MLPs. No hypernetwork. Configured under `learning_params.amor_params` (`policy_hidden_layer_sizes`, `value_hidden_layer_sizes`).

Shared infrastructure lives in `src/moplayground/moppo/`:
- `factory.py` — `make_morlax_networks`, `make_amor_networks`, and their inference-fn factories (`make_hypernetwork_inference_fn`, `make_amor_inference_fn`).
- `losses.py` — `compute_morlax_loss`, `compute_amor_loss`, plus `MORLAXNetworkParams` / `AMORNetworkParams`.
- `acting.py` — shared `MultiObjectiveTransition`, `actor_step`, `generate_unroll`, `Evaluator`.
- `networks.py` — hypernet variants (`Hypernet`, `HypernetMLP`, `DualA2CHypernet`, etc.).

Dispatch happens in `learning/training.py::train_policy` and `learning/inference.py::load_mo_policy` based on `config.algorithm`. AMOR's inference fn natively takes the directive at call time (`policy(obs, directive, key)`); `load_mo_policy` returns a 2-arg `policy(obs, key)` with the tradeoff baked in for compatibility with `mm.eval.rollout_policy`. To switch the directive at runtime, use `make_amor_inference_fn` directly.

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
