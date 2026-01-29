"""Hopper environment."""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from moplayground.envs.generic.mobase import MultiObjectiveBase
from moplayground.envs.dmcontrol.interface import AntInterface

class MOAnt(MultiObjectiveBase):
    """Hopper environment."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        super().__init__(
            xml_path          = AntInterface.XML,
            env_params        = env_params,
            backend           = backend,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, qpos_key, qvel_key = self._split(rng, 3)
        qpos = self._np.hstack([
            AntInterface.DEFAULT_FF,
            AntInterface.DEFAULT_JT
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
        info['xposbefore'] = data.qpos[0]
        info['yposbefore'] = data.qpos[1]
        info['xposafter']  = data.qpos[0] + 0.01
        info['yposafter']  = data.qpos[1] + 0.01
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
        state.info['yposbefore'] = state.data.qpos[1]
        action = self._np.clip(
            self.params.action_scale * action, 
            -1.0,
            1.0
        )
        data = self._step_fn(state.data, action)
        state.info['xposafter'] = data.qpos[0]
        state.info['yposafter'] = data.qpos[1]
        
        done = self.fall_termination(data)
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
        data
    ):
        infinite_state = ~self._np.isfinite(data.qpos).all()
        flipped_over = data.sensordata[2] < 0.45
        return infinite_state | flipped_over

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        obs = self._np.concatenate([
            data.qpos[2:],
            data.qvel,
        ])
        return {
            'state': obs,
            'privileged_state': obs
        }
    
    @property
    def action_size(self):
        return self.mj_model.nu

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
            'vx'     : self.reward_vx(info),
            'vy'     : self.reward_vy(info)
        }
        return rewards
    
    def reward_alive(self):
        return 1.0
    
    def reward_energy(self, action):
        return 8.0 - 1.0 * self._np.square(action).sum()
    
    def reward_vx(self, info):
        vx = (info['xposafter'] - info['xposbefore']) / self.dt
        return vx
    
    def reward_vy(self, info):
        vy = (info['yposafter'] - info['yposbefore']) / self.dt
        return vy