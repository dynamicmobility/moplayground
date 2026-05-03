---
layout: default
title: "moplayground.moppo.moppo"
parent: "moplayground.moppo"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/moppo.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.moppo.moppo`
Proximal policy optimization training. 

See: https://arxiv.org/pdf/1707.06347.pdf 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/moppo.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sample_preferences`

```python
sample_preferences(
    key,
    it,
    sampling,
    K,
    warmup_frac,
    alpha,
    num_evals,
    num_envs,
    num_objs
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/moppo.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train`

```python
train(
    environment: brax.envs.base.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[Callable[[brax.base.System, jax.Array], Tuple[brax.base.System, brax.base.System]]] = None,
    learning_rate: float = 0.0001,
    entropy_cost: float = 0.0001,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    alpha: float = 1.0,
    warmup_frac: float = 0.0,
    sampling: str = 'dense',
    k: int = 4,
    network_factory: brax.training.types.NetworkFactory[moplayground.moppo.factory.MOPPONetworks] = <function make_moppo_networks at 0x797cd2801b20>,
    init_policy_params: dict = None,
    init_normalizer_params: dict = None,
    init_value_params: dict = None,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    num_evals: int = 1,
    eval_env: Optional[brax.envs.base.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    progress_fn: Callable[[int, Mapping[str, jax.Array]], NoneType] = <function <lambda> at 0x797cd2803420>,
    policy_params_fn: Callable[..., NoneType] = <function <lambda> at 0x797cd28034c0>,
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/moppo.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MOTrainingState`
Contains training state for the learner. 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MOTrainingState.__init__`

```python
__init__(
    optimizer_state: Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, Iterable[ForwardRef('ArrayTree')], Mapping[Any, ForwardRef('ArrayTree')]],
    params: moplayground.moppo.losses.MOPPONetworkParams,
    normalizer_params: brax.training.acme.running_statistics.RunningStatisticsState,
    env_steps: brax.training.types.UInt64
) → None
```









