# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks."""

from typing import Literal, Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen


import dataclasses
import functools
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.spectral_norm import SNDense
from flax import linen
import jax
import jax.numpy as jnp
from moplayground.moppo.networks import Hypernet, HypernetMLP, FakeHypernet
from moplayground.moppo.networks import ActorCriticHypernet, DualA2CHypernet

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer  = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardHypernetwork:
    init            : Callable[..., Any]
    apply           : Callable[..., Any]
    get_features    : Callable[..., Any]
    get_flat_mlps   : Callable[..., Any]

@flax.struct.dataclass
class MORLAXNetworks:
    hypernetwork                   : FeedForwardHypernetwork
    policy_network                 : networks.FeedForwardNetwork
    value_network                  : networks.FeedForwardNetwork
    parametric_action_distribution : distribution.ParametricDistribution


def make_hypernetwork_inference_fn(ppo_networks: MORLAXNetworks):
    """Creates params and inference function for the PPO agent."""

    def hypernetwork_inference_fn(
        params: types.Params, directive: jax.Array, single_policy: bool = False, deterministic: bool = False
    ) -> types.Policy:
        normalizer_params, hypernet_params = params
        policy_network = ppo_networks.policy_network
        policy_params = ppo_networks.hypernetwork.apply(hypernet_params, directive)[0]

        if len(directive.shape) == 1:
            policy_apply = policy_network.apply
        else:
            policy_apply = jax.vmap(policy_network.apply, in_axes=(None, 0, 0))
        parametric_action_distribution = ppo_networks.parametric_action_distribution
       

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            param_subset = (normalizer_params, policy_params)  # normalizer and policy params
            logits = policy_apply(*param_subset, observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                'log_prob'   : log_prob,
                'raw_action' : raw_actions,
            }

        return policy

    return hypernetwork_inference_fn


def make_morlax_networks(
    observation_size          : types.ObservationSize,
    action_size               : int,
    num_objectives            : int,
    hypersize                 : tuple,
    key                       : jax.Array,
    target_policy_params      : dict = None,
    target_value_params       : dict = None,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes : Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes  : Sequence[int] = (256,) * 5,
    activation                : networks.ActivationFn = linen.swish,
    policy_obs_key            : str = 'state',
    value_obs_key             : str = 'state',
    distribution_type         : Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type            : Literal['scalar', 'log'] = 'scalar',
    init_noise_std            : float = 1.0,
    state_dependent_std       : bool = False,
    hypertype                 : str = 'MLP',
    num_features              : int = 8
) -> MORLAXNetworks:
    """Make PPO networks with preprocessor."""
    parametric_action_distribution: distribution.ParametricDistribution
    if distribution_type == 'normal':
        parametric_action_distribution = distribution.NormalDistribution(
            event_size=action_size
        )
    elif distribution_type == 'tanh_normal':
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
    else:
        raise ValueError(
            f'Unsupported distribution type: {distribution_type}. Must be one'
            ' of "normal" or "tanh_normal".'
        )
    policy_network = networks.make_policy_network(
        param_size                 = parametric_action_distribution.param_size,
        obs_size                   = observation_size,
        preprocess_observations_fn = preprocess_observations_fn,
        hidden_layer_sizes         = policy_hidden_layer_sizes,
        activation                 = activation,
        obs_key                    = policy_obs_key,
        distribution_type          = distribution_type,
        noise_std_type             = noise_std_type,
        init_noise_std             = init_noise_std,
        state_dependent_std        = state_dependent_std,
        # layer_norm                 = True
    )
    
    value_network = networks.make_value_network(
        obs_size                   = observation_size,
        preprocess_observations_fn = preprocess_observations_fn,
        hidden_layer_sizes         = value_hidden_layer_sizes,
        activation                 = activation,
        obs_key                    = value_obs_key,
    )

    if target_policy_params is None:
        target_policy_params = policy_network.init(key)
    if target_value_params is None:
        target_value_params = value_network.init(key)

    hypernetwork = make_hypernetwork(
        observation_size   = observation_size,
        num_objectives     = num_objectives,
        target_policy_dict = target_policy_params,
        target_value_dict  = target_value_params,
        hypersize          = hypersize,
        hypertype          = hypertype,
        policy_obs_key     = policy_obs_key,
        num_features       = num_features
    )

    return MORLAXNetworks(
        hypernetwork                   = hypernetwork,
        policy_network                 = policy_network,
        value_network                  = value_network,
        parametric_action_distribution = parametric_action_distribution,
    )

def make_hypernetwork(
    observation_size   : int,
    num_objectives     : int,
    target_policy_dict : dict,
    hypersize          : tuple,
    hypertype          : str = 'MLP',
    policy_obs_key     : str = 'state',
    num_features       : int = 8,
    target_value_dict  : dict = None,
):
    obs_dim = networks._get_obs_state_size(observation_size, policy_obs_key)
    if hypertype == 'MLP':
        hypernet = HypernetMLP(
            target_model_dict = target_policy_dict,
            num_objs          = num_objectives,
            obs_dim           = obs_dim,
            hypersize         = hypersize
        )
    elif hypertype == 'affine':
        hypernet = Hypernet(
            target_model_dict = target_policy_dict,
            num_objs          = num_objectives,
            obs_dim           = obs_dim,
            hypersize         = hypersize,
            num_features      = num_features
        )
    elif hypertype == 'ActorCritic':
        hypernet = DualA2CHypernet(
            target_policy_dict = target_policy_dict,
            target_value_dict  = target_value_dict,
            num_objs           = num_objectives,
            obs_dim            = obs_dim,
            hypersize          = hypersize,
            num_features       = num_features
        )        
    elif hypertype == 'fake':
        print('WARNING. Fake Hypernetwork in use.')
        hypernet = FakeHypernet(
            target_model_dict = target_policy_dict,
            num_objs          = num_objectives,
            obs_dim           = obs_dim,
            hypersize         = hypersize
        )
    else:
        raise Exception(f'Invalid hypertype {hypertype}')
    
    dummy_pref = jnp.zeros(num_objectives)
    def init(key):
        return hypernet.init(key, dummy_pref)
    
    def apply(params, prefs):
        if len(prefs.shape) == 1:
            prefs = prefs / jnp.sum(prefs)
        else:
            prefs = prefs / jnp.sum(prefs, axis=1)[:, jnp.newaxis]
        mlps, flat_mlps, features = hypernet.apply(params, prefs)
        return mlps, flat_mlps, features
    
    def get_mlps(params, prefs):
        return apply(params, prefs)[0]
    
    def get_flat_mlps(params, prefs):
        return apply(params, prefs)[1]
    
    def get_features(params, prefs):
        return apply(params, prefs)[2]
    
    wrapped_hypernet = FeedForwardHypernetwork(
        init            = init,
        apply           = get_mlps,
        get_flat_mlps = get_flat_mlps,
        get_features    = get_features
    )
    
    return wrapped_hypernet

@flax.struct.dataclass
class AMORNetworks:
    policy_network                 : networks.FeedForwardNetwork
    value_network                  : networks.FeedForwardNetwork
    parametric_action_distribution : distribution.ParametricDistribution


def _amor_normalize_and_concat(
    obs, directive, processor_params, preprocess_observations_fn, obs_key
):
    if isinstance(obs, Mapping):
        normalized = preprocess_observations_fn(
            obs[obs_key], networks.normalizer_select(processor_params, obs_key)
        )
    else:
        normalized = preprocess_observations_fn(obs, processor_params)
    return jnp.concatenate([normalized, directive], axis=-1)


def make_amor_policy_network(
    param_size                : int,
    obs_size                  : types.ObservationSize,
    num_objectives            : int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes        : Sequence[int] = (256, 256),
    activation                : networks.ActivationFn = linen.relu,
    kernel_init               : Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm                : bool = False,
    obs_key                   : str = 'state',
    distribution_type         : Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type            : Literal['scalar', 'log'] = 'scalar',
    init_noise_std            : float = 1.0,
    state_dependent_std       : bool = False,
) -> networks.FeedForwardNetwork:
    """AMOR policy network: input is concat(normalized_obs, raw_directive)."""
    if distribution_type == 'tanh_normal':
        policy_module = networks.MLP(
            layer_sizes = list(hidden_layer_sizes) + [param_size],
            activation  = activation,
            kernel_init = kernel_init,
            layer_norm  = layer_norm,
        )
    elif distribution_type == 'normal':
        policy_module = networks.PolicyModuleWithStd(
            param_size          = param_size,
            hidden_layer_sizes  = hidden_layer_sizes,
            activation          = activation,
            kernel_init         = kernel_init,
            layer_norm          = layer_norm,
            noise_std_type      = noise_std_type,
            init_noise_std      = init_noise_std,
            state_dependent_std = state_dependent_std,
        )
    else:
        raise ValueError(
            f'Unsupported distribution type: {distribution_type}. Must be one'
            ' of "normal" or "tanh_normal".'
        )

    def apply(processor_params, policy_params, obs, directive):
        x = _amor_normalize_and_concat(
            obs, directive, processor_params, preprocess_observations_fn, obs_key
        )
        return policy_module.apply(policy_params, x)

    obs_dim = networks._get_obs_state_size(obs_size, obs_key)
    augmented_dim = obs_dim + num_objectives
    dummy_input = jnp.zeros((1, augmented_dim))

    def init(key):
        return policy_module.init(key, dummy_input)

    return networks.FeedForwardNetwork(init=init, apply=apply)


def make_amor_value_network(
    obs_size                  : types.ObservationSize,
    num_objectives            : int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes        : Sequence[int] = (256, 256),
    activation                : networks.ActivationFn = linen.relu,
    obs_key                   : str = 'state',
) -> networks.FeedForwardNetwork:
    """AMOR value network: input is concat(normalized_obs, raw_directive),
    output is a vector of length ``num_objectives`` — one expected return per
    objective. The directive-scalarized advantage is formed in the loss."""
    value_module = networks.MLP(
        layer_sizes = list(hidden_layer_sizes) + [num_objectives],
        activation  = activation,
        kernel_init = jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, value_params, obs, directive):
        x = _amor_normalize_and_concat(
            obs, directive, processor_params, preprocess_observations_fn, obs_key
        )
        return value_module.apply(value_params, x)

    obs_dim = networks._get_obs_state_size(obs_size, obs_key)
    augmented_dim = obs_dim + num_objectives
    dummy_input = jnp.zeros((1, augmented_dim))
    return networks.FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_input), apply=apply
    )


def make_amor_networks(
    observation_size          : types.ObservationSize,
    action_size               : int,
    num_objectives            : int,
    key                       : jax.Array,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes : Sequence[int] = (64,) * 2,
    value_hidden_layer_sizes  : Sequence[int] = (256,) * 5,
    activation                : networks.ActivationFn = linen.swish,
    policy_obs_key            : str = 'state',
    value_obs_key             : str = 'state',
    distribution_type         : Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type            : Literal['scalar', 'log'] = 'scalar',
    init_noise_std            : float = 1.0,
    state_dependent_std       : bool = False,
) -> AMORNetworks:
    """Make AMOR networks (tradeoff-conditioned policy + multi-objective value).

    Unlike MORLAX (which uses a hypernetwork to output policy weights per directive),
    AMOR's policy and value networks take the directive as part of their input,
    concatenated to the (normalized) observation.
    """
    if distribution_type == 'normal':
        parametric_action_distribution = distribution.NormalDistribution(
            event_size=action_size
        )
    elif distribution_type == 'tanh_normal':
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
    else:
        raise ValueError(
            f'Unsupported distribution type: {distribution_type}. Must be one'
            ' of "normal" or "tanh_normal".'
        )

    policy_network = make_amor_policy_network(
        param_size                 = parametric_action_distribution.param_size,
        obs_size                   = observation_size,
        num_objectives             = num_objectives,
        preprocess_observations_fn = preprocess_observations_fn,
        hidden_layer_sizes         = policy_hidden_layer_sizes,
        activation                 = activation,
        obs_key                    = policy_obs_key,
        distribution_type          = distribution_type,
        noise_std_type             = noise_std_type,
        init_noise_std             = init_noise_std,
        state_dependent_std        = state_dependent_std,
    )

    value_network = make_amor_value_network(
        obs_size                   = observation_size,
        num_objectives             = num_objectives,
        preprocess_observations_fn = preprocess_observations_fn,
        hidden_layer_sizes         = value_hidden_layer_sizes,
        activation                 = activation,
        obs_key                    = value_obs_key,
    )

    return AMORNetworks(
        policy_network                 = policy_network,
        value_network                  = value_network,
        parametric_action_distribution = parametric_action_distribution,
    )


def make_amor_inference_fn(amor_networks: AMORNetworks):
    """Creates an inference function for AMOR.

    Returned closure: ``amor_inference_fn(params, deterministic=False) -> policy(obs, directive, key)``.
    Unlike MORLAX's ``hypernetwork_inference_fn`` (which bakes the directive into the
    closure at construction), AMOR's policy takes the directive at call time, so the
    tradeoff can be changed at any step without rebuilding the policy.
    """

    def amor_inference_fn(
        params: types.Params, deterministic: bool = False
    ) -> Callable:
        normalizer_params, policy_params = params
        dist = amor_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            directive: jax.Array,
            key_sample: jax.Array,
        ) -> Tuple[types.Action, types.Extra]:
            logits = amor_networks.policy_network.apply(
                normalizer_params, policy_params, observations, directive
            )
            if deterministic:
                return dist.mode(logits), {}
            raw_actions = dist.sample_no_postprocessing(logits, key_sample)
            log_prob = dist.log_prob(logits, raw_actions)
            postprocessed_actions = dist.postprocess(raw_actions)
            return postprocessed_actions, {
                'log_prob'   : log_prob,
                'raw_action' : raw_actions,
            }

        return policy

    return amor_inference_fn