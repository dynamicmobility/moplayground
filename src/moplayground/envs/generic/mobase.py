from minimal_mjx.envs.generic.base import SwappableBase
from ml_collections import config_dict
from pathlib import Path
import jax
import numpy as np

class MultiObjectiveBase(SwappableBase):
    """Base environment that emits a vector-valued reward.

    Extends ``minimal_mjx.envs.generic.base.SwappableBase`` so concrete
    environments can be constructed against either NumPy or JAX backends.
    Per-step reward components are bucketed into objective groups defined by
    ``env_params.reward.optimization.objectives`` (one entry per output
    dimension of the reward vector) and an optional set of
    ``shared_objectives`` that are added to every dimension.

    Args:
        xml_path: Path to the MuJoCo XML model.
        env_params: ConfigDict with at least ``reward.weights``,
            ``reward.optimization.objectives`` (list of lists of reward keys,
            one list per objective dimension), and
            ``reward.optimization.shared_objectives`` (list of reward keys
            added to every objective).
        backend: ``'jnp'`` for JAX (training), ``'np'`` for NumPy (eval).
        num_free: Number of free joints in the model. Forwarded to
            ``SwappableBase``.
    """

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
        """Combine per-key rewards into a vector reward plus updated metrics.

        Each entry of ``self.objectives`` maps to one component of the
        returned reward vector, computed as a weighted sum of the listed
        per-key rewards. Shared objectives are then added to every
        component.

        Args:
            rewards: Mapping from reward key to scalar reward for the
                current step.
            metrics: Existing metrics dict to extend.

        Returns:
            ``(reward, metrics)`` where ``reward`` has shape
            ``(len(self.objectives),)`` and ``metrics`` is the updated
            metrics dict.
        """
        metrics = self.get_metrics(metrics, rewards)
        
        weights = self.params.reward.weights
        mo_reward = self._np.array([])
        for rew_objs in self.objectives:
            reward = 0
            for key in rew_objs:
                reward += weights[key] * rewards[key]
            mo_reward = self._np.hstack([mo_reward, reward])
        shared_reward = {k: rewards[k] * weights[k] for k in self.shared_objectives}
        reward = mo_reward + sum(shared_reward.values())
        
        return reward, metrics
    

class Multi2SingleObjective:
    """Wrap a multi-objective env to expose a scalar reward.

    Replaces the vector reward from the wrapped environment with the inner
    product ``reward · weighting``, so the wrapped env can be plugged into
    standard single-objective PPO. All other attributes/methods are
    delegated to the underlying ``env`` via ``__getattr__``.

    Args:
        env: A ``MultiObjectiveBase`` (or compatible) environment whose
            ``reset``/``step`` return states with vector rewards.
        weighting: Per-objective weights, length must match the env's
            reward dimension.
    """

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