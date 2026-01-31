import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax import traverse_util
from typing import Literal, Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
import numpy as np
from typing import Sequence

class MLP(nn.Module):
    obs_dim: int                
    hidden_sizes: Sequence[int] 
    action_dim: int             

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.obs_dim)

        # Hidden layers
        for h in self.hidden_sizes:
            # x = nn.LayerNorm()(x)
            x = nn.Dense(h)(x)
            x = nn.swish(x)

        # Output layer
        x = nn.Dense(self.action_dim)(x)
        return x

def flatten_model(target_params):
    flat_params_list = []
    num_params = 0
    target_policy_info = []

    def flatten_dict(d, prefix=""):
        items = []
        for k, v in d.items():
            if isinstance(v, (dict, FrozenDict)):
                items.extend(flatten_dict(v, prefix + k + "/"))
            else:
                items.append((prefix + k, v))
        items.sort()
        return items

    for name, param in flatten_dict(target_params):
        shape = param.shape
        target_policy_info.append((name, shape))
        num_params += np.prod(shape)
        flat_params_list.append(param.reshape(-1))

    flat_init = jnp.concatenate(flat_params_list)

    return flat_init, num_params, target_policy_info


class Hypernet(nn.Module):
    target_model_dict: nn.Module
    num_objs: int
    obs_dim: int
    hypersize: tuple
    num_features: int = 8
    W_variance: float = 0.0

    def setup(self):
        # Init target policy params
        target_params = self.target_model_dict['params']

        num_params = 0
        target_policy_info = []

        _, num_params, target_policy_info = flatten_model(target_params)

        # Save as immutable attributes
        self.target_policy_info = tuple(target_policy_info)
        self.num_params = num_params

        # Hypernetwork MLP
        self.pref_mlp = MLP(
            obs_dim      = self.num_objs,
            hidden_sizes = self.hypersize,
            action_dim   = self.num_features
        )
        
        flat_init, num_params, target_policy_info = flatten_model(target_params)
        # print(target_policy_info)
        # target_policy_info = [('hidden_0/bias', (64,)), ('hidden_0/kernel', (17, 64)), ('hidden_1/bias', (64,)), ('hidden_1/kernel', (64, 64)), ('hidden_2/bias', (12,)), ('hidden_2/kernel', (64, 12))]


        self.target_policy_info = tuple(target_policy_info)
        self.num_params = num_params

        # Hypernet params
        self.W = self.param(
            "W",
            lambda key: (
                jax.random.uniform(key, (self.num_features, self.num_params), minval=-1, maxval=1)
                * self.W_variance
            ),
        )
        self.b = self.param("b", lambda key: flat_init)

    def __call__(self, pref, single=False):
        features = self.pref_mlp(pref)  # (..., num_features)
        flat_params = jnp.dot(features, self.W) + self.b  # (..., num_params)

        # Reshape back into dict
        sorted_params = {}
        cnt = 0
        for name, shape in self.target_policy_info:
            size = np.prod(shape)
            sorted_params[name] = flat_params[..., cnt:cnt+size].reshape((-1,) + shape)
            cnt += size

        nested_params = traverse_util.unflatten_dict({
            tuple(k.split("/")): v if not single else v[0] for k, v in sorted_params.items()
        })
        nested_params = FrozenDict(nested_params)

        # Wrap under "params" collection for Flax
        return flat_params, {"params": nested_params}
    
    
class HypernetMLP(nn.Module):
    target_model_dict: dict
    num_objs: int
    obs_dim: int
    hypersize: tuple

    def setup(self):
        # Init target policy params
        target_params = self.target_model_dict['params']

        num_params = 0
        target_policy_info = []

        _, num_params, target_policy_info = flatten_model(target_params)
        
        # Save as immutable attributes
        self.target_policy_info = tuple(target_policy_info)
        self.num_params = num_params

        # Hypernetwork MLP
        self.mlp = MLP(
            obs_dim      = self.num_objs,
            hidden_sizes = self.hypersize,
            action_dim   = self.num_params
        )

    def __call__(self, pref, single=False):
        flat_params = self.mlp(pref)

        # Reshape back into dict
        sorted_params = {}
        cnt = 0
        for name, shape in self.target_policy_info:
            size = np.prod(shape)
            sorted_params[name] = flat_params[..., cnt:cnt+size].reshape((-1,) + shape)
            cnt += size

        nested_params = traverse_util.unflatten_dict({
            tuple(k.split("/")): v if not single else v[0] for k, v in sorted_params.items()
        })
        nested_params = FrozenDict(nested_params)

        # Wrap under "params" collection for Flax
        return flat_params, {"params": nested_params}

class FakeHypernet(nn.Module):
    target_model_dict: dict
    num_objs: int
    obs_dim: int
    hypersize: tuple

    def setup(self):
        # Init target policy params
        target_params = self.target_model_dict['params']

        num_params = 0
        target_policy_info = []

        flat_params, num_params, target_policy_info = flatten_model(target_params)
        
        # Save as immutable attributes
        self.target_policy_info = tuple(target_policy_info)
        self.num_params = num_params
        self.flat_params = self.param(
            'flat_params',
            lambda key: flat_params
        )

    def __call__(self, pref, single=False):
        flat_params = self.flat_params
        if not single:
            flat_params = jnp.repeat(
                flat_params[jnp.newaxis, :], 
                pref.shape[0], 
                axis=0
            )
        
        # Reshape back into dict
        sorted_params = {}
        cnt = 0
        for name, shape in self.target_policy_info:
            size = np.prod(shape)
            sorted_params[name] = flat_params[..., cnt:cnt+size].reshape((-1,) + shape)
            cnt += size

        nested_params = traverse_util.unflatten_dict({
            tuple(k.split("/")): v if not single else v[0] for k, v in sorted_params.items()
        })
        nested_params = FrozenDict(nested_params)

        # Wrap under "params" collection for Flax
        return flat_params, {"params": nested_params}
    

class ActorCriticHypernet(nn.Module):
    target_policy_dict: dict
    target_value_dict: dict
    num_objs: int
    obs_dim: int
    hypersize: tuple = (128, 128)
    num_features: int = 8
    W_variance: float = 0.0

    def setup(self):
        # Init target policy params
        target_policy_params = self.target_policy_dict['params']
        target_value_params  = self.target_value_dict['params']

        target_policy_info = []
        target_value_info  = []

        policy_flat_init, num_policy_params, target_policy_info = flatten_model(target_policy_params)
        value_flat_init, num_value_params, target_value_info    = flatten_model(target_value_params)
        
        # Save as immutable attributes
        self.target_policy_info = tuple(target_policy_info)
        self.target_value_info  = tuple(target_value_info)
        self.num_policy_params  = num_policy_params
        self.num_value_params   = num_value_params
        self.num_params         = num_policy_params + num_value_params

        # Hypernetwork MLP
        self.mlp = MLP(
            obs_dim      = self.num_objs,
            hidden_sizes = self.hypersize,
            action_dim   = self.num_features
        )

        # Hypernet params
        self.W = self.param(
            "W",
            lambda key: (
                jax.random.uniform(key, (self.num_features, self.num_params), minval=-1, maxval=1)
                * self.W_variance
            ),
        )
        self.b = self.param(
            "b",
            lambda key: jnp.hstack((policy_flat_init, value_flat_init))
        )

    def unflatten_params(self, flat_params, target_network, single):
        # Reshape back into dict
        sorted_params = {}
        cnt = 0
        for name, shape in target_network:
            size = np.prod(shape)
            sorted_params[name] = flat_params[..., cnt:cnt+size].reshape((-1,) + shape)
            cnt += size

        nested_params = traverse_util.unflatten_dict({
            tuple(k.split("/")): v if not single else v[0] for k, v in sorted_params.items()
        })
        nested_params = FrozenDict(nested_params)

        # Wrap under "params" collection for Flax
        return {"params": nested_params}

    def __call__(self, pref, single=False):
        features = self.mlp(pref)
        flat_params = jnp.dot(features, self.W) + self.b

        policy_flat_params = flat_params[:self.num_policy_params]
        policy_sort_params = self.unflatten_params(
            flat_params    = policy_flat_params,
            target_network = self.target_policy_info,
            single         = single
        )

        value_flat_params  = flat_params[-self.num_value_params:]
        value_sort_params  = self.unflatten_params(
            flat_params    = value_flat_params,
            target_network = self.target_value_info,
            single         = single
        )

        return policy_sort_params, value_sort_params
    
    
class DualA2CHypernet(nn.Module):
    target_policy_dict: dict
    target_value_dict: dict
    num_objs: int
    obs_dim: int
    hypersize: tuple = (128, 128)
    num_features: int = 8
    W_variance: float = 0.0

    def setup(self):
        # Init target policy params
        target_policy_params = self.target_policy_dict['params']
        target_value_params  = self.target_value_dict['params']

        target_policy_info = []
        target_value_info  = []

        policy_flat_init, num_policy_params, target_policy_info = flatten_model(target_policy_params)
        value_flat_init, num_value_params, target_value_info    = flatten_model(target_value_params)
        
        # Save as immutable attributes
        self.target_policy_info = tuple(target_policy_info)
        self.target_value_info  = tuple(target_value_info)
        self.num_policy_params  = num_policy_params
        self.num_value_params   = num_value_params

        # Hypernetwork MLP
        self.policy_mlp = MLP(
            obs_dim      = self.num_objs,
            hidden_sizes = self.hypersize,
            action_dim   = self.num_features
        )
        self.value_mlp = MLP(
            obs_dim      = self.num_objs,
            hidden_sizes = self.hypersize,
            action_dim   = self.num_features
        )

        # Hypernet params
        self.policy_W = self.param(
            "policy_W",
            lambda key: (
                jax.random.uniform(key, (self.num_features, self.num_policy_params), minval=-1, maxval=1)
                * self.W_variance
            ),
        )
        self.policy_b = self.param(
            "policy_b",
            lambda key: policy_flat_init
        )
        
        # Hypernet params
        self.value_W = self.param(
            "value_W",
            lambda key: (
                jax.random.uniform(key, (self.num_features, self.num_value_params), minval=-1, maxval=1)
                * self.W_variance
            ),
        )
        self.value_b = self.param(
            "value_b",
            lambda key: value_flat_init
        )

    def unflatten_params(self, flat_params, target_network, single):
        # Reshape back into dict
        sorted_params = {}
        cnt = 0
        for name, shape in target_network:
            size = np.prod(shape)
            sorted_params[name] = flat_params[..., cnt:cnt+size].reshape((-1,) + shape)
            cnt += size

        nested_params = traverse_util.unflatten_dict({
            tuple(k.split("/")): v if not single else v[0] for k, v in sorted_params.items()
        })
        nested_params = FrozenDict(nested_params)

        # Wrap under "params" collection for Flax
        return {"params": nested_params}

    def __call__(self, pref, single=False):
        policy_features = self.policy_mlp(pref)
        value_features  = self.value_mlp(pref)
        policy_flat_params = jnp.dot(policy_features, self.policy_W) + self.policy_b
        value_flat_params  = jnp.dot(value_features,  self.value_W) + self.value_b

        policy_sort_params = self.unflatten_params(
            flat_params    = policy_flat_params,
            target_network = self.target_policy_info,
            single         = single
        )

        value_sort_params  = self.unflatten_params(
            flat_params    = value_flat_params,
            target_network = self.target_value_info,
            single         = single
        )

        return policy_sort_params, value_sort_params

def count_params(params):
    """Count the total number of parameters in a Flax model."""
    return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))