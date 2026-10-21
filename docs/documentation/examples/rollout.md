# Rolling out a policy

`scripts/rollout.py` runs one environment for a fixed duration and saves a video and
a plot of every reward term. It drives the environment with either a trained policy
conditioned on a tradeoff, or a dummy controller that needs no checkpoint.

```bash
# Trained policy at a random tradeoff
python -m scripts.rollout <save_dir>/<name>/config.yaml

# Trained policy at a chosen tradeoff (one weight per objective)
python -m scripts.rollout <save_dir>/<name>/config.yaml --tradeoff 1 0 0

# Dummy controllers: no policy, no checkpoint needed
python -m scripts.rollout config/mocheetah.yaml --inference_fn zero
python -m scripts.rollout config/mocheetah.yaml --inference_fn random
```

The video and plot are written to `output/videos/<env>-rollout.mp4` and
`output/videos/<env>-reward.pdf`.

## Command-line options

| Argument | Default | Meaning |
| --- | --- | --- |
| `config_path` | *(required)* | The YAML config. For a trained policy, the `config.yaml` saved in its run directory. |
| `--inference_fn` | `policy` | `policy` loads the latest checkpoint; `zero` or `random` use a dummy controller |
| `--tradeoff` | random | Objective weights for the policy, e.g. `--tradeoff 0.5 0.5`. Drawn uniformly from the simplex if omitted. |
| `--T` | `5.0` | Rollout length in seconds |
| `--camera` | `track` | Camera to render from |
| `--seed` | none | Seed NumPy's generator, which picks the random tradeoff and the initial state's key |

## Policy or dummy controller

Both are *inference functions*: callables `(obs, rng) -> (action, extras)`, the
signature Brax and minimal_mjx use.

```python
if inference_fn_type == 'policy':
    inference_fn = jax.jit(load_policy(cfg, tradeoff))
else:
    inference_fn = mm.make_dummy_inference_fn(env=env, mode=inference_fn_type)
```

**Trained policy.** `load_policy` restores the newest checkpoint in `cfg.run_dir` with
the network parameters of the config's algorithm. For `morlax` and `amor` it calls
[`load_mo_policy`][moplayground.learning.inference.load_mo_policy], which fixes the
tradeoff inside the returned function, so the policy takes the standard two
arguments. For `ppo` it uses `minimal_mjx`'s `load_policy`, and the tradeoff is
ignored. Actions are the mean of the policy distribution (deterministic).

**Dummy controller.** `make_dummy_inference_fn` (from minimal_mjx) returns all-zero
actions (`zero`) or uniform random actions in [-1, 1] (`random`). Use it to check a
new environment's dynamics, rewards and termination before training anything; the
config does not need a run directory.

!!! tip "Check that a new environment is JAX-compatible"
    Training always uses the `jnp` backend. Before training a new environment, roll
    it out once with `backend: jnp` in the config and `--inference_fn zero`. Code that
    runs under NumPy but cannot be traced by JAX (Python `if` on array values,
    in-place array writes) fails here in seconds rather than after a training job has
    started.

## Backend

The rollout uses the backend named in the config. `backend: np` steps MuJoCo's C
bindings directly, which is fast for a single environment and allows breakpoints and
prints inside `reset` and `step`. JAX is pinned to the CPU (`JAX_PLATFORMS=cpu`),
since only the small policy network is evaluated with JAX.

## Varying the tradeoff at runtime

The function returned by `load_mo_policy` has its tradeoff fixed. To change the
tradeoff from step to step with AMOR, build the inference function directly; it takes
the tradeoff as an argument:

```python
make_policy, params = mop.load_make_amor_inference_fn(
    cfg.amor_network_params, cfg.num_objectives, run_dir=cfg.run_dir
)
policy = make_policy(params=(params[0], params[1]), deterministic=True)
action, _ = policy(obs, tradeoff, key)       # tradeoff may change on every call
```

To evaluate a run over many tradeoffs at once, use
[`get_morlax_fronts`][moplayground.eval.pareto.get_morlax_fronts] or
[`get_amor_fronts`][moplayground.eval.pareto.get_amor_fronts], which run batched JAX
rollouts and return the achieved returns for every checkpoint.

## Full script

??? example "scripts/rollout.py"

    ```python
    --8<-- "scripts/rollout.py"
    ```
