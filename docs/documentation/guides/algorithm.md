# Writing an algorithm

MORLAX and AMOR are two instances of one pattern: PPO in which every parallel
environment is assigned a tradeoff, the policy is conditioned on that tradeoff, and the
vector reward is scalarized by it in the loss. This page states the contract a new
multi-objective trainer must meet, lists the building blocks it can reuse, and shows
how to register it with `scripts/train.py`.

The quickest starting point is to copy `src/moplayground/moppo/amor.py`, the simpler
of the two, and change the networks and the loss.

## The contract

`mm.train` (from minimal_mjx) drives every algorithm through a **setup function**:

```python
def setup_my_algo(train_fn_params: dict, network_factory_params: dict, resumed):
    network_factory = functools.partial(make_my_networks, **network_factory_params)
    train_fn = functools.partial(
        my_algo.train, **dict(train_fn_params),
        network_factory = network_factory,
        wrap_env_fn     = mop.mo_wrapper,
    )
    return train_fn, network_factory
```

`mm.train` then:

1. records the checkpoint network config with Brax's `checkpoint.network_config`,
   which reads the keyword arguments bound into `network_factory`. They must be
   JSON-serializable, and the factory must accept `observation_size`, `action_size`
   and `preprocess_observations_fn`;
2. calls `train_fn(environment=env, eval_env=eval_env, progress_fn=..., policy_params_fn=...)`;
3. expects it to return `(make_inference_fn, params, metrics)`.

Inside `train`, the algorithm must call the two callbacks:

| Callback | When | Arguments |
| --- | --- | --- |
| `policy_params_fn` | Once before training and after every epoch | `(current_step, make_inference_fn, params)`. `mm.train` writes `params` as the checkpoint `<run_dir>/<step>/`. |
| `progress_fn` | After every evaluation | `(current_step, metrics)`. For [`MOProgress`][moplayground.utils.plotting.MOProgress], `metrics` must hold `'reward'`, the per-objective episode returns of shape `(num_eval_envs, num_objectives)`, and `'tradeoff'`, the tradeoffs that produced them. |

`resumed` is always `None` from `scripts/train.py`, which does not resume
multi-objective runs.

## Building blocks

Everything below is shared by MORLAX and AMOR and can be reused unchanged.

**Environment wrapper.** [`mo_wrapper`][moplayground.learning.training.mo_wrapper]
vmaps the environment and adds
[`MultiObjectiveEpisodeWrapper`][moplayground.learning.wrappers.MultiObjectiveEpisodeWrapper]
(episode length, truncation, vector-valued episode returns) and auto-reset. The
wrapped environment's `state.reward` has shape `(num_envs, num_objectives)`.

**Tradeoff sampling.**
[`sample_preferences`][moplayground.moppo.morlax.sample_preferences] draws one tradeoff
per environment according to `sampling_params`, and returns an array of shape
`(1, num_envs, num_objectives)`.

**Unrolls.** [`generate_unroll`][moplayground.moppo.acting.generate_unroll] steps the
environment with a policy and a fixed batch of tradeoffs, and returns a
[`MultiObjectiveTransition`][moplayground.moppo.acting.MultiObjectiveTransition]:
the Brax transition with a vector `reward` and a `tradeoff` field.

```python
tradeoff = sample_preferences(key_pref, it, sampling, k, warmup_frac, alpha,
                               num_evals, num_envs, num_objectives)[0]
policy = make_policy(params, tradeoff)            # policy(obs, key) -> (action, extras)
state, data = acting.generate_unroll(
    env, state, policy, tradeoff, key, unroll_length,
    extra_fields=('truncation', 'episode_metrics', 'episode_done'),
)
```

**Losses.** Two ways to use the vector reward:

- *Scalarize first*: `rewards = jnp.sum(data.tradeoff * data.reward, axis=-1)`, then a
  standard scalar critic and GAE. This is what
  [`compute_morlax_loss`][moplayground.moppo.losses.compute_morlax_loss] does.
- *Vector critic*: predict one value per objective and compute per-objective
  advantages with [`compute_mo_gae`][moplayground.moppo.losses.compute_mo_gae], then
  scalarize the advantages by the tradeoff. This is what
  [`compute_amor_loss`][moplayground.moppo.losses.compute_amor_loss] does.

**Evaluation.** [`Evaluator`][moplayground.moppo.acting.Evaluator] rolls out
`num_eval_envs` episodes at tradeoffs drawn uniformly from the simplex, and returns
exactly the `'reward'` and `'tradeoff'` metrics `MOProgress` expects:

```python
evaluator = acting.Evaluator(
    eval_env       = eval_env,
    num_objs       = num_objectives,
    eval_policy_fn = make_eval_policy,   # (params, tradeoffs) -> policy(obs, key)
    num_eval_envs  = num_eval_envs,
    episode_length = episode_length,
    action_repeat  = action_repeat,
    key            = eval_key,
)
metrics = evaluator.run_evaluation(params, training_metrics)
progress_fn(current_step, metrics)
```

**Networks.** [`factory`][moplayground.moppo.factory] builds the Brax policy and value
networks. [`make_amor_networks`][moplayground.moppo.factory.make_amor_networks] shows a
factory for networks that take the tradeoff as an input, and
[`make_morlax_networks`][moplayground.moppo.factory.make_morlax_networks] one that
generates network weights from it. Hypernetwork modules are in
[`networks`][moplayground.moppo.networks].

## Registering the algorithm

1. **Setup function.** Add `setup_my_algo` (above) next to `setup_morlax` and
   `setup_amor` in `src/moplayground/learning/training.py`, and export it in
   `src/moplayground/__init__.py`.
2. **Config.** Add a `my_algo_network_params` section under `learning_params`, and a
   matching property on [`MOConfig`][moplayground.utils.config.MOConfig] (see
   [Adding a key](configuration.md#adding-a-key)).
3. **Training script.** Add a case to `get_handle_params` in `scripts/train.py`:

    ```python
    case 'my_algo':
        return mop.setup_my_algo, cfg.ppo_params | cfg.sampling_params, cfg.my_algo_network_params
    ```

    `get_progress` already returns `MOProgress` for every algorithm except `ppo`.

4. **Inference.** Add a case to
   [`load_mo_policy`][moplayground.learning.inference.load_mo_policy] that restores a
   checkpoint and returns a `policy(obs, key)` with the tradeoff fixed, and to
   `load_policy` in `scripts/rollout.py`. For Pareto-front evaluation, pass your own
   `make_policy` and `load_params_fn` to
   [`compute_fronts`][moplayground.eval.pareto.compute_fronts], which is
   algorithm-agnostic.
5. **Tests.** Add the algorithm to `ALGORITHMS` and `NETWORK_PROPERTY` in
   `tests/unit/test_configs_dir.py` and to `test_dispatch` in
   `tests/unit/test_scripts.py`. The GPU tier (`pytest -m gpu`) trains a tiny run of
   each algorithm in `ALGORITHMS` (`tests/gpu/conftest.py`) end to end.

Then set `algorithm: my_algo` in a config and train as usual:

```bash
python -m scripts.train config/my_config.yaml
```
