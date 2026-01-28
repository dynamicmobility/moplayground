"""Hopper environment."""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from minimal_mjx.envs.generic.base import MultiObjectiveBase
from moplayground.dmcontrol.interface import WalkerInterface

class MOWalker(MultiObjectiveBase):
    """Hopper environment."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        super().__init__(
            xml_path          = WalkerInterface.XML,
            env_params        = env_params,
            backend           = backend,
            num_free          = 3
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._np.hstack([
            WalkerInterface.DEFAULT_FF,
            WalkerInterface.DEFAULT_JT
        ])
        qvel = self._np.zeros(self.mj_model.nv)
        ctrl = qpos[self.num_free:].copy()

        data = self._data_init_fn(
            qpos         = qpos,
            qvel         = qvel,
            ctrl         = ctrl,
            time         = 0.0,
            xfrc_applied = self._np.zeros((self._mj_model.nbody, 6)),
        )
        parent_state = super().reset(
            rng            = rng,
            data           = data,
            history_length = self.params.history_length
        )
        info = {}
        info['posbefore'] = data.qpos[0]
        info['posafter']  = data.qpos[0] + 0.01
        info = parent_state.info | info

        done = self._np.array(0.0)
        rewards = self.reward_function(
            data   = data,
            action = ctrl,
            info   = info,
            done   = False,
        )
        reward, metrics = self.get_reward_and_metrics(rewards, {})
        
        obs = self._get_obs(data, parent_state.info)
        return self._state_init_fn(data, obs, reward, done, metrics, info)
    
    def state_vector(self, data):
        return self._np.hstack([data.qpos, data.qvel])

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info['posbefore'] = state.data.qpos[0]
        action = self._np.clip(
            self.params.action_scale * action, 
            -1.0,
             1.0
        )
        data = self._step_fn(state.data, action)
        state.info['posafter'] = data.qpos[0]
        
        done = self.termination(data)
        rewards = self.reward_function(
            data   = data,
            action = action,
            info   = state.info,
            done   = done
        )
        reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
        obs = self._get_obs(
            data,
            state.info
        )
        done = done.astype(float)
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)
    
    def termination(
        self,
        data: mjx.Data,
    ):
        height, ang = data.qpos[1:3]
        good_height = (height > 0.8) & (height < 2.0)
        good_ang    = (ang > -1.0) & (ang < 1.0)
        return ~(good_height & good_ang)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        obs = self._np.concatenate([
            data.qpos[1:],
            self._np.clip(
                data.qvel,
                -10.0,
                10.0
            ),
        ])
        return {
            'state': obs,
            'privileged_state': obs
        }
    
    @property
    def action_size(self):
        return 6

    def reward_function(
        self,
        data,
        action,
        info,
        done
    ):
        rewards = {
            'alive'  : self.reward_alive(),
            'energy' : self.reward_energy(action),
            'run'    : self.reward_run(info),
            'done'   : self.reward_done(done)
        }
        return rewards
    
    def reward_alive(self):
        return 1.0
    
    def reward_energy(self, action):
        return 4.0 - 1.0 * self._np.square(action).sum()
    
    def reward_run(self, info):
        vx = (info['posafter'] - info['posbefore']) / self.dt
        return vx
    
    def reward_done(self, done):
        return self._np.array(done)
