from minimal_mjx.envs.generic.base import SwappableBase
from ml_collections import config_dict
from pathlib import Path
import jax
import numpy as np

class MultiObjectiveBase(SwappableBase):
    def __init__(self,
        xml_path          : Path,
        env_params        : config_dict.ConfigDict,
        backend           : str = 'jnp',
        num_free          : int = 3
    ):
        self.objectives        = env_params.reward.optimization.objectives
        self.shared_objectives = env_params.reward.optimization.shared_objectives
        super().__init__(
            xml_path   = xml_path,
            env_params = env_params,
            backend    = backend,
            num_free   = num_free
        )
        
    def get_reward_and_metrics(
        self,
        rewards : jax.Array,
        metrics : dict
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        metrics = self.get_metrics(metrics, rewards)
        
        weights = self.params.reward.weights
        mo_reward = self._np.array([])
        for key in self.objectives:
            mo_reward = self._np.hstack([mo_reward, weights[key] * rewards[key]])
        shared_reward = {k: rewards[k] * weights[k] for k in self.shared_objectives}
        reward = mo_reward + sum(shared_reward.values())
        
        return reward, metrics
    

class Multi2SingleObjective:
    def __init__(self, env, weighting):
        self.env = env
        self.weighting = weighting

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(reward=self.env._np.sum(state.reward * self.env._np.array(self.weighting)))

    def step(self, state, action):
        state = self.env.step(state, action)
        return state.replace(reward=self._np.sum(state.reward * self.env._np.array(self.weighting)))
    
    def __getattr__(self, name):
        """
        Called only if `name` is not found on self.
        Delegate lookup to the base object.
        """
        return getattr(self.env, name)