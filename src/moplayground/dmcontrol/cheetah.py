"""Hopper environment."""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from minimal_mjx.envs.generic.base import MultiObjectiveBase
from moplayground.dmcontrol.interface import CheetahInterface

class MOHalfCheetah(MultiObjectiveBase):
    """Hopper environment."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        super().__init__(
            xml_path          = CheetahInterface.XML,
            env_params        = env_params,
            backend           = backend,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, qpos_key, qvel_key = self._split(rng, 3)
        qpos = self._np.hstack([
            CheetahInterface.DEFAULT_FF,
            CheetahInterface.DEFAULT_JT
        ])
        qvel = self._np.zeros(self.mj_model.nv)
        ctrl = self._np.zeros(self.mj_model.nu)
        
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
        info['xposbefore'] = 0.0
        info['xposafter']  = 0.01
        info['ang']        = 0.0
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
        state.info['xposbefore'] = state.data.qpos[0]
        action = self._np.clip(
            self.params.action_scale * action, 
            -1.0,
            1.0
        )
        data = self._step_fn(state.data, action)
        state.info['xposafter'] = data.qpos[0]
        state.info['ang']       = data.qpos[2]
        
        done = self.fall_termination(state.info)
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
    
    def fall_termination(
        self,  
        info: dict
    ):
        return self._np.array(
            ~(abs(info['ang']) < self._np.deg2rad(50))
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        obs = self._np.concatenate([
            data.qpos[1:],
            self._np.clip(data.qvel, -10.0, 10.0),
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
        reward_run = (info['xposafter'] - info['xposbefore']) / self.dt
        return self._np.min(self._np.array([4.0, reward_run]))
    
    def reward_done(self, done):
        return self._np.array(done)
