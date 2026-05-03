---
layout: default
title: "moplayground.moppo.factory"
parent: "moplayground.moppo"
grand_parent: API Reference
---

<!-- markdownlint-disable -->

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `moplayground.moppo.factory`
PPO networks. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_hypernetwork_inference_fn`

```python
make_hypernetwork_inference_fn(
    ppo_networks: moplayground.moppo.factory.MOPPONetworks
)
```

Creates params and inference function for the PPO agent. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_moppo_networks`

```python
make_moppo_networks(
    observation_size: Union[int, Mapping[str, Union[Tuple[int, ...], int]]],
    action_size: int,
    num_objectives: int,
    hypersize: tuple,
    key: jax.Array,
    target_policy_params: dict = None,
    target_value_params: dict = None,
    preprocess_observations_fn: brax.training.types.PreprocessObservationFn = <function identity_observation_preprocessor at 0x797dacbfbd80>,
    policy_hidden_layer_sizes: Sequence[int] = (32, 32, 32, 32),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    activation: Callable[[jax.Array], jax.Array] = <PjitFunction of <function silu at 0x797df4637380>>,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    hypertype: str = 'MLP',
    num_features: int = 8
) → MOPPONetworks
```

Make PPO networks with preprocessor. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_hypernetwork`

```python
make_hypernetwork(
    observation_size: int,
    num_objectives: int,
    target_policy_dict: dict,
    hypersize: tuple,
    hypertype: str = 'MLP',
    policy_obs_key: str = 'state',
    num_features: int = 8,
    target_value_dict: dict = None
)
```






---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_mo_value_network`

```python
make_mo_value_network(
    obs_size: Union[int, Mapping[str, Union[Tuple[int, ...], int]]],
    num_objectives: int,
    preprocess_observations_fn: brax.training.types.PreprocessObservationFn = <function identity_observation_preprocessor at 0x797dacbfbd80>,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: Callable[[jax.Array], jax.Array] = <jax._src.custom_derivatives.custom_jvp object at 0x797df49d1fd0>,
    obs_key: str = 'state'
) → FeedForwardNetwork
```

Creates a value network. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeedForwardHypernetwork`
FeedForwardHypernetwork(init: Callable[..., Any], apply: Callable[..., Any], get_features: Callable[..., Any], get_flat_mlps: Callable[..., Any]) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FeedForwardHypernetwork.__init__`

```python
__init__(
    init: Callable[..., Any],
    apply: Callable[..., Any],
    get_features: Callable[..., Any],
    get_flat_mlps: Callable[..., Any]
) → None
```









---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MOPPONetworks`
MOPPONetworks(hypernetwork: moplayground.moppo.factory.FeedForwardHypernetwork, policy_network: brax.training.networks.FeedForwardNetwork, value_network: brax.training.networks.FeedForwardNetwork, parametric_action_distribution: brax.training.distribution.ParametricDistribution) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MOPPONetworks.__init__`

```python
__init__(
    hypernetwork: moplayground.moppo.factory.FeedForwardHypernetwork,
    policy_network: brax.training.networks.FeedForwardNetwork,
    value_network: brax.training.networks.FeedForwardNetwork,
    parametric_action_distribution: brax.training.distribution.ParametricDistribution
) → None
```









