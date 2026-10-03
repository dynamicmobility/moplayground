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
    ppo_networks: moplayground.moppo.factory.MORLAXNetworks
)
```

Creates params and inference function for the PPO agent. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_morlax_networks`

```python
make_morlax_networks(
    observation_size: Union[int, Mapping[str, Union[Tuple[int, ...], int]]],
    action_size: int,
    num_objectives: int,
    hypersize: tuple,
    key: jax.Array,
    target_policy_params: dict = None,
    target_value_params: dict = None,
    preprocess_observations_fn: brax.training.types.PreprocessObservationFn = <function identity_observation_preprocessor at 0x7ffe8b64ae80>,
    policy_hidden_layer_sizes: Sequence[int] = (32, 32, 32, 32),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    activation: Callable[[jax.Array], jax.Array] = <PjitFunction of <function silu at 0x7fff375a74c0>>,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    hypertype: str = 'MLP',
    num_features: int = 8
) → MORLAXNetworks
```

Make PPO networks with preprocessor. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_amor_policy_network`

```python
make_amor_policy_network(
    param_size: int,
    obs_size: Union[int, Mapping[str, Union[Tuple[int, ...], int]]],
    num_objectives: int,
    preprocess_observations_fn: brax.training.types.PreprocessObservationFn = <function identity_observation_preprocessor at 0x7ffe8b64ae80>,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: Callable[[jax.Array], jax.Array] = <jax._src.custom_derivatives.custom_jvp object at 0x7fff3755a930>,
    kernel_init: Callable[..., Any] = <function variance_scaling.<locals>.init at 0x7ffa3c976d40>,
    layer_norm: bool = False,
    obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False
) → FeedForwardNetwork
```

AMOR policy network: input is concat(normalized_obs, raw_directive). 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L319"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_amor_value_network`

```python
make_amor_value_network(
    obs_size: Union[int, Mapping[str, Union[Tuple[int, ...], int]]],
    num_objectives: int,
    preprocess_observations_fn: brax.training.types.PreprocessObservationFn = <function identity_observation_preprocessor at 0x7ffe8b64ae80>,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: Callable[[jax.Array], jax.Array] = <jax._src.custom_derivatives.custom_jvp object at 0x7fff3755a930>,
    obs_key: str = 'state'
) → FeedForwardNetwork
```

AMOR value network: input is concat(normalized_obs, raw_directive), output is a vector of length ``num_objectives`` — one expected return per objective. The directive-scalarized advantage is formed in the loss. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L350"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_amor_networks`

```python
make_amor_networks(
    observation_size: Union[int, Mapping[str, Union[Tuple[int, ...], int]]],
    action_size: int,
    num_objectives: int,
    key: jax.Array,
    preprocess_observations_fn: brax.training.types.PreprocessObservationFn = <function identity_observation_preprocessor at 0x7ffe8b64ae80>,
    policy_hidden_layer_sizes: Sequence[int] = (64, 64),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    activation: Callable[[jax.Array], jax.Array] = <PjitFunction of <function silu at 0x7fff375a74c0>>,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False
) → AMORNetworks
```

Make AMOR networks (tradeoff-conditioned policy + multi-objective value). 

Unlike MORLAX (which uses a hypernetwork to output policy weights per directive), AMOR's policy and value networks take the directive as part of their input, concatenated to the (normalized) observation. 


---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_amor_inference_fn`

```python
make_amor_inference_fn(amor_networks: moplayground.moppo.factory.AMORNetworks)
```

Creates an inference function for AMOR. 

Returned closure: ``amor_inference_fn(params, deterministic=False) -> policy(obs, directive, key)``. Unlike MORLAX's ``hypernetwork_inference_fn`` (which bakes the directive into the closure at construction), AMOR's policy takes the directive at call time, so the tradeoff can be changed at any step without rebuilding the policy. 


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

## <kbd>class</kbd> `MORLAXNetworks`
MORLAXNetworks(hypernetwork: moplayground.moppo.factory.FeedForwardHypernetwork, policy_network: brax.training.networks.FeedForwardNetwork, value_network: brax.training.networks.FeedForwardNetwork, parametric_action_distribution: brax.training.distribution.ParametricDistribution) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MORLAXNetworks.__init__`

```python
__init__(
    hypernetwork: moplayground.moppo.factory.FeedForwardHypernetwork,
    policy_network: brax.training.networks.FeedForwardNetwork,
    value_network: brax.training.networks.FeedForwardNetwork,
    parametric_action_distribution: brax.training.distribution.ParametricDistribution
) → None
```









---

<a href="https://github.com/dynamicmobility/moplayground/blob/main/src/moplayground/moppo/factory.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AMORNetworks`
AMORNetworks(policy_network: brax.training.networks.FeedForwardNetwork, value_network: brax.training.networks.FeedForwardNetwork, parametric_action_distribution: brax.training.distribution.ParametricDistribution) 

<a href="https://github.com/dynamicmobility/moplayground/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AMORNetworks.__init__`

```python
__init__(
    policy_network: brax.training.networks.FeedForwardNetwork,
    value_network: brax.training.networks.FeedForwardNetwork,
    parametric_action_distribution: brax.training.distribution.ParametricDistribution
) → None
```









